import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from skywork_tokenizer import SkyworkTokenizerAPI
from skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL
from config import *  # expects LM_MODEL_NAME, SKYWORK_MODEL_NAME, STEP_TOKEN, MAX_NEW_TOKENS, NUM_TOKEN_CANDIDATES

import json
from torch.utils.data import Dataset


#********************************
#                  Config / Setup
#********************************

lm_device = torch.device("cuda:0")
prm_device = torch.device("cuda:1")


#********************************
#                     Load models
#********************************

# Load language model (policy model)
tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LM_MODEL_NAME,
    torch_dtype=torch.bfloat16
).to(lm_device).eval()

# Load PRM (Process Reward Model)
prm_tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, STEP_TOKEN)
prm = PRM_MODEL.from_pretrained(SKYWORK_MODEL_NAME).to(prm_device).eval()


#********************************
#                     Dataset util
#********************************

class PRM800k(Dataset):
    """
    A custom PyTorch Dataset class to load the PRM800k dataset from a local .jsonl file.
    """
    def __init__(self, jsonl_path, size):
        self.samples = []
        print(f"Loading {size} samples from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == size:
                    break
                if line.strip():
                    data = json.loads(line)
                    question = data["question"]["problem"]
                    answer = data["question"]["pre_generated_steps"]
                    self.samples.append((question, answer))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, answer = self.samples[idx]
        return question, answer


#********************************
#        Greedy PRM token picker
#********************************

def _pick_next_token(
    logits: torch.Tensor,
    generated_ids,
    q_inputs,
    num_token_candidates: int,
):
    """
    Given current LM logits and the tokens generated so far, greedily select the
    next token by scanning the top-k LM candidates and choosing the one that
    maximizes the PRM reward at the last step.
    """
    # Base sequence from original PRM input_ids (batch size 1 assumed)
    base_ids = q_inputs.data["input_ids"][0].tolist()

    # Construct: [original question+steps] + generated_ids + [0] + step_token_ids
    mod_inputs = {}
    mod_inputs["input_ids"] = [
        base_ids + generated_ids + [0] + prm_tokenizer.step_token_ids
    ]
    mod_inputs["input_ids"] = torch.tensor(
        mod_inputs["input_ids"], device=prm_device
    )

    # Full attention over the modified sequence
    mod_inputs["attention_mask"] = torch.ones_like(
        mod_inputs["input_ids"], device=prm_device
    )

    # Extend reward flags: copy old, add zeros for new tokens + step_token_ids
    base_reward_flags = q_inputs.data["reward_flags"][0]  # (seq_len,)
    num_new_tokens = len(generated_ids) + 1 + len(prm_tokenizer.step_token_ids)
    new_flags = torch.zeros(num_new_tokens, device=prm_device)
    mod_inputs["reward_flags"] = torch.cat((base_reward_flags, new_flags))
    # Flag the last position as rewardable
    mod_inputs["reward_flags"][-1] = 1

    # Location of the candidate token to be replaced in the sequence
    mod_loc = len(mod_inputs["input_ids"][0]) - len(prm_tokenizer.step_token_ids) - 1

    # Consider top-k LM candidates
    k = min(num_token_candidates, logits.size(-1))
    _, indices = torch.topk(logits, k, dim=-1)
    indices = indices[0]  # remove batch dimension

    arg_max = None
    best_reward = None

    for token_id in indices:
        # Plug candidate token into the modified sequence
        mod_inputs["input_ids"][0, mod_loc] = token_id

        # Evaluate PRM reward for this candidate
        out = prm(**mod_inputs, return_probs=True)
        # out[2] is assumed shape (batch=1, seq_len)
        modified_reward = out[2][0][mod_inputs["reward_flags"].bool()]

        # Greedy selection based on final step reward
        if best_reward is None or modified_reward[-1] > best_reward[-1]:
            best_reward = modified_reward
            arg_max = int(token_id)

    return arg_max, best_reward


#********************************
#     Public callable entrypoint
#********************************

def greedy_prm_optimal_ids(
    question: str,
    steps,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_token_candidates: int = NUM_TOKEN_CANDIDATES,
    verbose: bool = False,
):
    """
    Run greedy PRM-guided token selection for a given (question, steps) pair.

    Returns
    -------
    best_prefix_ids : list[int]
        The token IDs (in the LM's vocab) corresponding to the best-scoring prefix
        under the PRM, according to the end-of-prefix reward.
    """
    #********************************
    #         Prepare LM / PRM inputs
    #********************************

    # For LM (generation) – we only care about the final token position + cache
    m_inputs = prm_tokenizer.prepare_steps(question, steps)
    m_inputs.to(lm_device)

    # For PRM (reward evaluation)
    q_inputs = prm_tokenizer.prepare_steps(question, steps)
    q_inputs.to(prm_device)

    with torch.no_grad():  # Pre-fill LM cache
        out = model(**m_inputs, use_cache=True)
        logits = out.logits[:, -1, :]
        cache = out.past_key_values

    # Optional: baseline PRM reward on original steps (not strictly needed)
    with torch.no_grad():
        out = prm(**q_inputs, return_probs=True)
        default_reward = out[2][0][q_inputs.data["reward_flags"][0].bool()]

    if verbose:
        print("default stepwise reward:", default_reward)

    #********************************
    #       Greedy discrete optimizer
    #********************************

    generated_ids = []
    prefix_end_rewards = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_id, modified_reward = _pick_next_token(
                logits, generated_ids, q_inputs, num_token_candidates
            )
            end_reward = modified_reward[-1].item()
            prefix_end_rewards.append(end_reward)

            if verbose:
                print(str(modified_reward.tolist()) + "\n############################")

            # Append and (optionally) show current continuation
            generated_ids.append(next_id)
            if verbose:
                continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(continuation + "\n----------------------------")

            # Feed chosen token back into LM with cached kvs
            next_token = torch.tensor([[next_id]], device=lm_device)
            out = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=cache,
            )
            logits = out.logits[:, -1, :]
            cache = out.past_key_values

            # Early stopping on EOS / heuristic sentence terminators
            if (
                next_id == tokenizer.eos_token_id
                or next_id == 13
                or next_id == 624
                or next_id == 382
            ):
                break

    #********************************
    #      Pick best prefix by reward
    #********************************

    if len(prefix_end_rewards) > 0:
        # Index of best prefix (0-based): corresponds to prefix length = best_idx + 1
        best_idx = int(torch.tensor(prefix_end_rewards).argmax().item())
        best_prefix_len = best_idx + 1
    else:
        # Edge case: no tokens generated at all
        best_prefix_len = 0

    best_prefix_ids = generated_ids[:best_prefix_len]

    if verbose:
        print("prefix_end_rewards:", prefix_end_rewards)
        print("best_prefix_len:", best_prefix_len)
        print("best_prefix_ids:", best_prefix_ids)
        print("best continuation:", tokenizer.decode(best_prefix_ids, skip_special_tokens=True))

    # The callable returns just the optimal generated IDs
    return best_prefix_ids


#********************************
#              Example usage
#********************************

if __name__ == "__main__":
    # Example: use PRM800k dataset to grab a sample
    dataset = PRM800k("phase2_train.jsonl", DATA_SUBSET_SIZE)
    question, steps = dataset[2]  # pick any index you like

    best_ids = greedy_prm_optimal_ids(question, steps, verbose=True)
    print("Final best ids:", best_ids)