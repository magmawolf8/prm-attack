import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from skywork_tokenizer import SkyworkTokenizerAPI
from skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL
from config import *

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
#           Example problem setup
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

prm800k_dataset = PRM800k("phase2_train.jsonl", DATA_SUBSET_SIZE)
question, steps = prm800k_dataset[0]

#question = (
#    "A class of 50 students has various hobbies. 10 like to bake, 5 like to play "
#    "basketball, and the rest like to either play video games or play music. "
#    "How many like to play video games if the number that like to play music is "
#    "twice the number that prefer playing basketball?"
#)
#steps = [
#    "To find out how many students like to play video games, let's start with the information given: "
#    "There are 50 students in total. 10 students like to bake. 5 students like to play basketball. "
#    "The number of students who like to play music is twice the number of students who like to play basketball.",
#    "First, we find out how many students like to play music: "
#    "\\[ \\text{Number of students who like to play music} = 2 \\times 5 = 10 \\]",
#    "Now, we subtract the number of students who like to bake and play basketball from the total number of students to "
#    "find out how many students like to play video games: "
#    "\\[ \\text{Number of students who like to play video games} = \\text{Total students} - "
#    "(\\text{Bake students} + \\text{Basketball students}) \\] "
#    "\\[ \\text{Number of students who like to play video games} = 50 - (10 + 5) \\] "
#    "\\[ \\text{Number of students who like to play video games} = 50 - 15 \\] "
#    "\\[ \\text{Number of students who like to play video games} = 35 \\]",
#    "So, there are 35 students who like to play video games."
#]


#********************************
#         Prepare LM / PRM inputs
#********************************

# For LM (generation)
m_inputs = prm_tokenizer.prepare_steps(question, steps)
m_inputs.to(lm_device)

# For PRM (reward evaluation)
q_inputs = prm_tokenizer.prepare_steps(question, steps)
q_inputs.to(prm_device)

with torch.no_grad():  # Pre-fill LM cache
    out = model(**m_inputs, use_cache=True)
    logits = out.logits[:, -1, :]
    cache = out.past_key_values

with torch.no_grad():  # Baseline PRM reward on original steps
    out = prm(**q_inputs, return_probs=True)
    default_reward = out[2][q_inputs.data["reward_flags"].bool()]

print("default stepwise reward:", default_reward)


#********************************
#        Greedy PRM token picker
#********************************

def pick_next_token(logits: torch.Tensor, generated_ids):
    """
    Given current LM logits and the tokens generated so far, greedily select the
    next token by scanning the top-k LM candidates and choosing the one that
    maximizes the PRM reward at the last step.
    """
    # Base sequence from original PRM input_ids
    mod_inputs = {}
    base_ids = q_inputs.data["input_ids"].tolist()[0]

    # Construct: [original question+steps] + generated_ids + [0] + step_token_ids
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
    num_new_tokens = len(generated_ids) + 1 + len(prm_tokenizer.step_token_ids)
    mod_inputs["reward_flags"] = torch.cat(
        (
            q_inputs.data["reward_flags"][0],
            torch.zeros(num_new_tokens, device=prm_device),
        )
    )
    # Flag the last position as rewardable
    mod_inputs["reward_flags"][-1] = 1

    # Location of the candidate token to be replaced in the sequence
    mod_loc = len(mod_inputs["input_ids"][0]) - len(prm_tokenizer.step_token_ids) - 1

    # Consider top-k LM candidates
    _, indices = torch.topk(logits, NUM_TOKEN_CANDIDATES, dim=-1)
    indices = indices[0]  # remove batch dimension

    arg_max = None
    best_reward = None

    for token_id in indices:
        # Plug candidate token into the modified sequence
        mod_inputs["input_ids"][0, mod_loc] = token_id

        # Evaluate PRM reward for this candidate
        out = prm(**mod_inputs, return_probs=True)
        modified_reward = out[2][0][mod_inputs["reward_flags"].bool()]

        # Greedy selection based on final step reward
        if best_reward is None or modified_reward[-1] > best_reward[-1]:
            best_reward = modified_reward
            arg_max = token_id

    return arg_max, best_reward


#********************************
#       Greedy discrete optimizer
#********************************

generated_ids = []
prefix_end_rewards = []

with torch.no_grad():
    for _ in range(MAX_NEW_TOKENS):
        next_id, modified_reward = pick_next_token(logits, generated_ids)
        end_reward = modified_reward[-1].item()
        prefix_end_rewards.append(end_reward)

        print(str(modified_reward.tolist()) + "\n############################")

        # Append and show current continuation
        generated_ids.append(next_id)
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
print("prefix_end_rewards:", prefix_end_rewards)
print("best_prefix_len:", best_prefix_len)

#********************************
#       Final PRM evaluation / IO
#********************************

ids = q_inputs.data["input_ids"].tolist()
ids = ids[0] + best_prefix_ids + prm_tokenizer.step_token_ids

mod_inputs = {}
mod_inputs["input_ids"] = [ids]
mod_inputs["input_ids"] = torch.tensor(
    mod_inputs["input_ids"], device=prm_device
)

mod_inputs["attention_mask"] = torch.ones_like(
    mod_inputs["input_ids"], device=prm_device
)

mod_inputs["reward_flags"] = torch.cat(
    (
        q_inputs.data["reward_flags"][0],
        torch.zeros(len(best_prefix_ids) + len(prm_tokenizer.step_token_ids), device=prm_device),
    )
)
mod_inputs["reward_flags"][-1] = 1

with torch.no_grad():  # Recompute PRM reward on final sequence
    out = prm(**mod_inputs, return_probs=True)
    regen_modified_reward = out[2][0][mod_inputs["reward_flags"].bool()]

print("regenerated, modified stepwise reward", regen_modified_reward)

continuation = tokenizer.decode(ids, skip_special_tokens=True)
print(continuation)
print(repr(continuation))

print(best_prefix_ids)