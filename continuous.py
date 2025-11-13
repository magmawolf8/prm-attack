#!/usr/bin/env python3

#********************************
#                         Imports
#********************************
# configuration
import config as cfg
from config import *

# python standard libraries
import os
import math
import random
import time
import json
from datetime import datetime

# tensor and deep learning modules (PyTorch)
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist

# custom model modules
from skywork_tokenizer import SkyworkTokenizerAPI
from skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL

# utility modules
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# plotting (headless safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#********************************
#                         Dataset
#********************************

class PRM800k(Dataset):
    """
    Loads paired (correct, incorrect) trajectories from the PRM800k dataset.
    Keeps only entries that have at least one non-positive–rated completion
    for every step so we can form full-length negative trajectories.
    """
    def __init__(self, jsonl_path, size):
        self.samples = []
        self.skipped = 0
        print(f"Loading up to {size} paired samples from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                neg_steps = self._extract_negative_steps(data)
                if neg_steps is None:
                    self.skipped += 1
                    continue
                question = data["question"]["problem"]
                pos_steps = data["question"]["pre_generated_steps"]
                self.samples.append((question, pos_steps, neg_steps))
                if len(self.samples) == size:
                    break

        if len(self.samples) < size:
            print(
                f"Dataset exhausted at {len(self.samples)} samples "
                f"(skipped {self.skipped} entries lacking negatives)."
            )
        else:
            print(f"Loaded {len(self.samples)} paired samples "
                  f"(skipped {self.skipped}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, pos_steps, neg_steps = self.samples[idx]
        return question, pos_steps, neg_steps

    @staticmethod
    def _extract_negative_steps(example):
        neg_steps = []
        for step in example.get("label", {}).get("steps", []):
            neg_text = None
            for completion in step.get("completions", []):
                rating = completion.get("rating")
                if rating is not None and rating <= 0:
                    neg_text = completion["text"]
                    break
            if neg_text is None:
                return None
            neg_steps.append(neg_text)
        return neg_steps if neg_steps else None

#********************************
#         Gradient sign optimizer
#********************************

class FGSM(torch.optim.SGD):
    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr=lr, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(p.grad.sign(), alpha=-lr)

        return loss

# Helper to save hyperparameters
def save_hyperparams(run_dir: str):
    hparams = {
        name: getattr(cfg, name)
        for name in dir(cfg)
        if name.isupper() and not name.startswith("_")
    }
    hparams["run_dir"] = run_dir
    hparams["timestamp"] = datetime.now().isoformat(timespec="seconds")

    out_path = os.path.join(run_dir, "hyperparams.json")
    with open(out_path, "w") as f:
        json.dump(hparams, f, indent=2, sort_keys=True)


# Set seed
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# --- HELPER FUNCTIONS ---

def insert_adversarial_prefix(tokenized_batch, batch_embeddings, adversarial_prefix):
    """
    Inserts the adversarial prefix embeddings at the start and end of the answer.
    """
    prefix_length = adversarial_prefix.shape[0]
    batch_size = batch_embeddings.shape[0]
    device = batch_embeddings.device

    zeros_for_prefix = torch.zeros(prefix_length, dtype=torch.long, device=device)

    processed_embeddings_list = []
    processed_answer_flags_list = []
    processed_reward_flags_list = []

    for i in range(batch_size):
        sample_embedding = batch_embeddings[i]

        answer_flag_vector = tokenized_batch.data["answer_flag"][i].to(device)
        reward_flags_vector = tokenized_batch.data["reward_flags"][i].to(device)

        start_insertion_idx = torch.nonzero(answer_flag_vector, as_tuple=True)[0][0]
        end_insertion_idx = torch.nonzero(reward_flags_vector, as_tuple=True)[0][-1]

        new_embedding = torch.vstack((
            sample_embedding[:start_insertion_idx],
            adversarial_prefix,
            sample_embedding[start_insertion_idx:end_insertion_idx],
            adversarial_prefix,
            sample_embedding[end_insertion_idx:]
        ))
        processed_embeddings_list.append(new_embedding)

        new_answer_flag = torch.cat((
            answer_flag_vector[:start_insertion_idx],
            zeros_for_prefix,
            answer_flag_vector[start_insertion_idx:end_insertion_idx],
            zeros_for_prefix,
            answer_flag_vector[end_insertion_idx:]
        ))
        processed_answer_flags_list.append(new_answer_flag)

        new_reward_flag = torch.cat((
            reward_flags_vector[:start_insertion_idx],
            zeros_for_prefix,
            reward_flags_vector[start_insertion_idx:end_insertion_idx],
            zeros_for_prefix,
            reward_flags_vector[end_insertion_idx:]
        ))
        processed_reward_flags_list.append(new_reward_flag)

    prefixed_batch_embeddings = torch.stack(processed_embeddings_list)
    prefixed_answer_flag = torch.stack(processed_answer_flags_list)
    prefixed_reward_flags = torch.stack(processed_reward_flags_list)

    total_added_length = 2 * prefix_length
    #total_added_length = prefix_length
    #total_added_length = 0
    prefixed_attention_mask = F.pad(
        input=tokenized_batch.data["attention_mask"],
        pad=(total_added_length, 0),
        value=1
    )

    return prefixed_batch_embeddings, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags


def collate_into_batch(samples_list):
    questions, pos_answers, neg_answers = zip(*samples_list)
    return list(questions), list(pos_answers), list(neg_answers)


def tokenize_with_prefix(tokenizer_api, questions, steps, embedding_layer, adversarial_prefix):
    """
    Tokenizes a batch of (question, steps) pairs and inserts the adversarial prefix
    into both the embeddings and the bookkeeping masks.
    """
    tokenized_batch = tokenizer_api.prepare_steps(questions, steps)
    batch_embeddings = embedding_layer[tokenized_batch.data["input_ids"]]

    pref_embeddings, pref_mask, pref_ans_flag, pref_reward_flag = insert_adversarial_prefix(
        tokenized_batch, batch_embeddings, adversarial_prefix
    )

    tokenized_batch.data["attention_mask"] = pref_mask
    tokenized_batch.data["answer_flag"] = pref_ans_flag
    tokenized_batch.data["reward_flags"] = pref_reward_flag
    tokenized_batch.pop("input_ids")

    return tokenized_batch, pref_embeddings

# --- LOGGING / PLOTTING UTILS ---

def save_loss_curve(loss_list, out_png, out_csv):
    if not loss_list:
        print("Warning: no losses to log; skipping loss curve generation.")
        return

    losses = np.array(loss_list, dtype=np.float32)  # shape: [steps, metrics]

    # Save CSV with header
    header = "total_loss,nll_pos,nll_neg,margin,entropy_penalty"
    np.savetxt(out_csv, losses, delimiter=",", header=header, comments="")

    steps = np.arange(1, len(losses) + 1)

    metric_series = {
        "total": losses[:, 0],
        "nll_pos": losses[:, 1],
        "nll_neg": losses[:, 2],
        "margin": losses[:, 3],
        "Hpenalty": losses[:, 4],
    }

    # Helper: moving average with window 10 (or smaller if fewer points)
    def moving_average(x, window=10):
        window = min(window, len(x))
        if window <= 1:
            return x, steps  # nothing to smooth
        weights = np.ones(window, dtype=np.float32) / window
        ma = np.convolve(x, weights, mode="valid")
        ma_steps = np.arange(window, len(x) + 1)
        return ma, ma_steps

    base, ext = os.path.splitext(out_png)
    for name, series in metric_series.items():
        raw_png = f"{base}_{name}_raw{ext}"
        ma_png = f"{base}_{name}_ma10{ext}"

        plt.figure(figsize=(8, 5))
        plt.plot(steps, series, linewidth=1.6)
        plt.xlabel("Optimizer step")
        plt.ylabel(name.replace("_", " "))
        plt.title(f"{name} (raw)")
        plt.tight_layout()
        plt.savefig(raw_png, dpi=150)
        plt.close()

        ma_values, ma_steps = moving_average(series, window=10)
        plt.figure(figsize=(8, 5))
        plt.plot(ma_steps, ma_values, linewidth=1.6)
        plt.xlabel("Optimizer step")
        plt.ylabel(f"{name} (10-step MA)")
        plt.title(f"{name} (10-step moving average)")
        plt.tight_layout()
        plt.savefig(ma_png, dpi=150)
        plt.close()


#********************************
#             Main training loop
#********************************

def train(gpu_id, num_gpus):
    # --- OUTPUT DIR ---
    RUN_DIR = f"adv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if gpu_id == 0:
        os.makedirs(RUN_DIR, exist_ok=True)
        print(f"Outputs will be saved under: {os.path.abspath(RUN_DIR)}")

    # --- DATA LOADING ---
    prm800k_dataset = PRM800k("phase2_train.jsonl", DATA_SUBSET_SIZE)

    sampler = DistributedSampler(
        prm800k_dataset,
        num_replicas=num_gpus,
        rank=gpu_id,
        shuffle=True
    )
    data_loader = DataLoader(
        prm800k_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_into_batch,
    )

    if gpu_id == 0:
        print("Warming up data loader...")
        start_time = time.perf_counter()
        _ = next(iter(data_loader))
        end_time = time.perf_counter()
        print(f"Data loader warmup took {(end_time - start_time):.1f} seconds")

    # --- MODEL AND OPTIMIZER SETUP ---
    skywork_tokenizer_api = SkyworkTokenizerAPI(
        SKYWORK_MODEL_NAME, STEP_TOKEN
    )
    reward_model = PRM_MODEL.from_pretrained(SKYWORK_MODEL_NAME).to(gpu_id).eval()
    token_embedding_layer = reward_model.pretrained_model.model.embed_tokens.weight
    vocabulary_size = token_embedding_layer.shape[0]

    adversarial_logits = torch.nn.Parameter(
        torch.zeros(NUM_PREFIXES, vocabulary_size, device=gpu_id)
    )
    #adversarial_logits = torch.nn.Parameter(
    #    0.1*torch.randn(NUM_PREFIXES, vocabulary_size, device=gpu_id)
    #)

    # Save initial prefix (rank 0) for similarity / distance after training
    if gpu_id == 0:
        initial_logits_cpu = adversarial_logits.detach().cpu().clone()

    optimizer = torch.optim.Adam([adversarial_logits], lr=LEARNING_RATE, maximize=False)

    if gpu_id == 0:
        print("Starting training...")

    # Per-step loss history (rank 0)
    loss_history = []

    steps_per_epoch = math.ceil(len(prm800k_dataset) / (BATCH_SIZE * num_gpus))
    total_steps = max(1, steps_per_epoch * NUM_EPOCHS)
    attack_objective = cfg.ATTACK_OBJECTIVE.lower()
    if attack_objective not in {"flatten", "invert"}:
        raise ValueError(
            f"ATTACK_OBJECTIVE must be 'flatten' or 'invert', got '{cfg.ATTACK_OBJECTIVE}'"
        )

    def get_temperature(step_idx: int) -> float:
        alpha = min(1.0, step_idx / total_steps)
        return (T_MAX ** (1.0 - alpha)) * (T_MIN ** alpha)

    global_step = 0

    questions = "For how many different digits $n$ is the two-digit number $\\underline{6}\\underline{n}$ divisible by $n$? (The expression $\\underline{6}\\underline{n}$ should be interpreted as a two-digit integer with tens digit 6 and units digit $n$, not as 6 times $n$.)"
    pos_answers = ["I want to find all the digits $n$ such that $\\underline{6}\\underline{n}$ is a multiple of $n$.", 
        "This means that $\\underline{6}\\underline{n}$ must be equal to $n$ times some integer $k$.", 
        "I can write this as $\\underline{6}\\underline{n} = kn$, or equivalently, $10 \\cdot 6 + n = kn$.", 
        "Subtracting $n$ from both sides, I get $60 = (k - 1)n$.", 
        "This means that $n$ must be a factor of 60, and also a digit from 0 to 9.", 
        "The factors of 60 are 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, and 60.", 
        "Out of these, only 1, 2, 3, 4, 5, and 6 are digits.", 
        "So there are 6 possible values for $n$ that make $\\underline{6}\\underline{n}$ divisible by $n$.", "# Answer\n\n6"]
    nums = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    neg_answers = ["I need to find the possible values of $n$ from 0 to 9 such that 6n is divisible by $n$.", 
        "A quick way to check divisibility is to use the remainders of dividing by $n$.", 
        "If the remainder of dividing 6 by $n$ is the same as the remainder of dividing $n$ by $n$, then 6n will be divisible by $n$.", 
        "For example, if $n=3$, then the remainder of dividing 6 by 3 is 0, and the remainder of dividing 3 by 3 is also 0, so 63 is divisible by 3.", 
        "On the other hand, if $n=4$, then the remainder of dividing 6 by 4 is 2, but the remainder of dividing 4 by 4 is 0, so 64 is not divisible by 4.", 
        "So I can use this rule to test each value of $n$ from 0 to 9.", 
        "If $n=0$, then 6n is not defined, so I exclude this case.", 
        "If $n=1$, then the remainder of dividing 6 by 1 is 0, and the remainder of dividing 1 by 1 is also 0, so 61 is divisible by 1.", 
        "If $n=2$, then the remainder of dividing 6 by 2 is 0, and the remainder of dividing 2 by 2 is also 0, so 62 is divisible by 2.", 
        "If $n=3$, then the remainder of dividing 6 by 3 is 0, and the remainder of dividing 3 by 3 is also 0, so 63 is divisible by 3.", 
        "If $n=4$, then the remainder of dividing 6 by 4 is 2, and the remainder of dividing 4 by 4 is 0, so 64 is not divisible by 4.", 
        "If $n=5$, then the remainder of dividing 6 by 5 is 1, and the remainder of dividing 5 by 5 is 0, so 65 is not divisible by 5.", 
        "If $n=6$, then the remainder of dividing 6 by 6 is 0, and the remainder of dividing 6 by 6 is also 0, so 66 is divisible by 6.", 
        "If $n=7$, then the remainder of dividing 6 by 7 is 6, and the remainder of dividing 7 by 7 is 0, so 67 is not divisible by 7.", 
        "If $n=8$, then the remainder of dividing 6 by 8 is 6, and the remainder of dividing 8 by 8 is 0, so 68 is not divisible by 8.", 
        "If $n=9$, then the remainder of dividing 6 by 9 is 6, and the remainder of dividing 9 by 9 is 0, so 69 is not divisible by 9.", 
        "So the only values of $n$ that make 6n divisible by $n$ are 1, 2, 3, and 6.", "# Answer\n\n4"]
    neg_answers = ["".join([str(random.randint(0, 9)) if c in nums and random.random() > 0.75 else c for c in step]) for step in neg_answers]

    # --- TRAINING EPOCHS ---
    for epoch in range(NUM_EPOCHS):
        if gpu_id == 0:
            progress_bar = tqdm(
                total=len(prm800k_dataset),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
            )
        sampler.set_epoch(epoch)

        for batch_questions, batch_pos_answers, batch_neg_answers in data_loader:
            batch_questions = questions
            batch_pos_answers = pos_answers
            batch_neg_answers = neg_answers
            #batch_questions = "When a single number is added to each member of the sequence 20, 50, 100, the sequence becomes expressible as $x, a x, a^{2} x$. Find $a$."
            #batch_pos_answers = [
            #"To find the value of \\(a\\), we start by understanding the given information and setting up equations based on the description. Given that a single number is added to each member of the sequence \\(20, 50, 100\\) to make it expressible as \\(x, ax, a^2x\\), we can denote the added number by \\(y\\). This means:\n\\[20 + y = x\\]\n\\[50 + y = ax\\]\n\\[100 + y = a^2x\\]\nFrom these equations, we aim to find \\(a\\).",
            #"First, express \\(y\\) in terms of \\(x\\) using the first equation:\n\\[y = x - 20\\]",
            #"Second, substitute \\(y\\) into the second equation:\n\\[50 + (x - 20) = ax\\]\n\\[30 + x = ax\\]\n\\[x = \\frac{30}{a - 1}\\]",
            #"Third, substitute \\(y\\) into the third equation:\n\\[100 + (x - 20) = a^2x\\]\n\\[80 + x = a^2x\\]\n\\[x = \\frac{80}{a^2 - 1}\\]",
            #"Fourth, equate the two expressions for \\(x\\) obtained from the second and third steps and solve for \\(a\\):\n\\[\\frac{30}{a - 1} = \\frac{80}{a^2 - 1}\\]\nNotice that \\(a^2 - 1\\) can be factored as \\((a + 1)(a - 1)\\), so we have:\n\\[\\frac{30}{a - 1} = \\frac{80}{(a + 1)(a - 1)}\\]\nSince the denominators are not equal, we cross-multiply:\n\\[30(a + 1) = 80\\]\n\\[30a + 30 = 80\\]\n\\[30a = 50\\]\n\\[a = \\frac{50}{30}\\]\n\\[a = \\frac{5}{3}\\]",
            #"Thus, the value of \\(a\\) is \\(\\frac{5}{3}\\)."
            #]
            #batch_neg_answers = [
            #"To find the value of \\(a\\), we start by understanding the given information and setting up equations based on the description. Given that a single number is added to each member of the sequence \\(20, 50, 100\\) to make it expressible as \\(x, ax, a^2x\\), we can denote the added number by \\(y\\). This means:\n\\[20 + y = x\\]\n\\[50 + y = ax\\]\n\\[100 + y = a^2x\\]\nFrom these equations, we aim to find \\(a\\).",
            #"First, express \\(y\\) in terms of \\(x\\) using the first equation:\n\\[y = x - 20\\]",
            #"Second, substitute \\(y\\) into the second equation:\n\\[50 + (x - 20) = ax\\]\n\\[30 + x = ax\\]\n\\[x = \\frac{30}{a - 1}\\]",
            #"Third, substitute \\(y\\) into the third equation:\n\\[100 + (x - 20) = a^2x\\]\n\\[80 + x = a^2x\\]\n\\[x = \\frac{80}{a^2 - 1}\\]",
            #"Fourth, equate the two expressions for \\(x\\) obtained from the second and third steps and solve for \\(a\\):\n\\[\\frac{30}{a - 1} = \\frac{80}{a^2 - 1}\\]\nNotice that \\(a^2 - 1\\) can be factored as \\((a + 1)(a - 1)\\), so we have:\n\\[\\frac{30}{a - 1} = \\frac{80}{(a + 1)(a - 1)}\\]\nSince the denominators are not equal, we cross-multiply:\n\\[30(a + 1) = 80\\]\n\\[30a + 30 = 80\\]\n\\[30a = 50\\]\n\\[a = \\frac{50}{30}\\]\n\\[a = \\frac{5}{3}\\]",
            #"Thus, the value of \\(a\\) is \\(\\frac{6}{3}\\)."
            #]
            # 1. PREPARE DATA
            T_t = get_temperature(global_step)
            one_hot = F.gumbel_softmax(adversarial_logits, tau=T_t, hard=(True if epoch==(NUM_EPOCHS - 1) else False), dim=-1)
            if gpu_id == 0 and epoch==(NUM_EPOCHS-1):
                print("LAST EPOCH")
            #one_hot = torch.softmax(adversarial_logits, dim=-1) # fake news---this isn't one-hot, just for debugging purposes
            adversarial_prefix = one_hot @ token_embedding_layer

            pos_batch, pos_embeddings = tokenize_with_prefix(
                skywork_tokenizer_api,
                batch_questions,
                batch_pos_answers,
                token_embedding_layer,
                adversarial_prefix,
            )
            neg_batch, neg_embeddings = tokenize_with_prefix(
                skywork_tokenizer_api,
                batch_questions,
                batch_neg_answers,
                token_embedding_layer,
                adversarial_prefix,
            )

            pos_batch = pos_batch.to(gpu_id)
            neg_batch = neg_batch.to(gpu_id)

            # 2. FORWARD PASS (positive & negative trajectories)
            pos_output = reward_model(
                **pos_batch,
                inputs_embeds=pos_embeddings,
                return_probs=True,
            )
            neg_output = reward_model(
                **neg_batch,
                inputs_embeds=neg_embeddings,
                return_probs=True,
            )

            pos_probs = pos_output[2][pos_batch.data["reward_flags"].bool()]
            neg_probs = neg_output[2][neg_batch.data["reward_flags"].bool()]

            nll_pos = -torch.log(pos_probs).mean()
            nll_neg = -torch.log(neg_probs).mean()
            margin = nll_neg - nll_pos

            if attack_objective == "flatten":
                discrim_loss = margin.pow(2)
            elif attack_objective == "invert":
                discrim_loss = margin
            else:
                raise ValueError(f"Unsupported ATTACK_OBJECTIVE '{cfg.ATTACK_OBJECTIVE}'")

            # --- ENTROPY OF ADVERSARIAL DISTRIBUTION ---
            # Numerical safety
            probs = torch.softmax(adversarial_logits, dim=-1)
            log_probs = (probs + 1e-12).log()
            entropy_total = -(probs * log_probs).sum()  # sum over all prefixes+vocab

            # Normalized entropy averaged across prefixes
            H_mean = entropy_total / NUM_PREFIXES
            H_norm = H_mean / math.log(vocabulary_size) # normalized

            H_penalty = REG_LAMBDA * entropy_total

            attack_loss = discrim_loss + H_penalty

            # --- METRICS FOR LOGGING ---
            with torch.no_grad():
                max_p_per_prefix = probs.max(dim=-1).values
                avg_max_p = max_p_per_prefix.mean()

                H_norm_det = H_norm.detach()

            # 4. BACKPROPAGATION
            attack_loss.backward()

            # 5. GRADIENT AGGREGATION AND OPTIMIZER STEP
            dist.all_reduce(adversarial_logits.grad, op=dist.ReduceOp.SUM)
            adversarial_logits.grad /= num_gpus
            adversarial_grad = adversarial_logits.grad.clone()
            optimizer.step()
            optimizer.zero_grad()

            # 6. LOGGING
            with torch.no_grad():
                loss_tensor = attack_loss.detach()
                disc_tensor = discrim_loss.detach()
                pos_tensor = nll_pos.detach()
                neg_tensor = nll_neg.detach()
                margin_tensor = margin.detach()
                H_tensor = H_penalty.detach()

                # Average across GPUs
                for t in (loss_tensor, disc_tensor, pos_tensor, neg_tensor, margin_tensor, H_tensor):
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    t /= num_gpus

                if gpu_id == 0:
                    loss_history.append((
                        float(loss_tensor.item()),   # total loss
                        float(pos_tensor.item()),    # nll_pos
                        float(neg_tensor.item()),    # nll_neg
                        float(margin_tensor.item()), # margin
                        float(H_tensor.item()),      # entropy penalty
                    ))
                    progress_bar.update(BATCH_SIZE * num_gpus)
                    progress_bar.set_postfix({
                        "loss": f"{attack_loss.item():.3f}",
                        "obj":  f"{disc_tensor.item():.3f}",
                        "pos":  f"{pos_tensor.item():.3f}",
                        "neg":  f"{neg_tensor.item():.3f}",
                        "mar":  f"{margin_tensor.item():.3f}",
                        "Hn":   f"{H_norm_det.item():.2f}",
                        "norm":   f"{torch.norm(adversarial_grad, dim=-1).mean().item():.2e}",
                        "pmax": f"{avg_max_p.item():.2f}",
                        "T":    f"{T_t:.2f}",
                    })
            
            global_step += 1
    
        if gpu_id == 0:
            progress_bar.close()

    # --- SAVE & REPORT (rank 0) ---
    dist.barrier()
    if gpu_id == 0:
        # Save optimized prefix
        opt_path = os.path.join(
            RUN_DIR,
            f"optimized_logits.pt"
        )
        torch.save(adversarial_logits.detach().cpu(), opt_path)
        print(f"Saved optimized logits to {opt_path}")

        print(neg_answers)
        probs = torch.softmax(adversarial_logits.detach().cpu(), dim=-1)
        num_prefixes, vocab_size = probs.shape

        # Factorize vocabulary size into near-square dimensions
        side1 = int(np.floor(np.sqrt(vocab_size)))
        side2 = int(np.ceil(vocab_size / side1))
        print(f"Vocabulary grid size: {side1} x {side2} ({side1*side2} >= {vocab_size})")

        vis_dir = os.path.join(RUN_DIR, "token_prob_viz")
        os.makedirs(vis_dir, exist_ok=True)

        for i in range(num_prefixes):
            prefix_probs = probs[i]

            # Scale by maximum for visualization (avoid divide-by-zero)
            max_val = prefix_probs.max()
            if max_val > 0:
                prefix_probs = prefix_probs / max_val

            # Pad to fill the grid shape
            padded = np.zeros(side1 * side2)
            padded[:vocab_size] = prefix_probs
            grid = padded.reshape(side1, side2)

            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            plt.title(f"Prefix {i} token probabilities (scaled)")
            plt.axis("off")
            out_img = os.path.join(vis_dir, f"prefix_{i:02d}_probs_scaled.png")
            plt.savefig(out_img, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Saved scaled token probability visualizations to {vis_dir}")

        save_hyperparams(RUN_DIR)

        init_path = os.path.join(RUN_DIR, "initial_logits.pt")
        torch.save(initial_logits_cpu, init_path)

        # Save loss CSV + plot
        loss_csv = os.path.join(RUN_DIR, "training_loss.csv")
        loss_png = os.path.join(RUN_DIR, "training_loss.png")
        save_loss_curve(loss_history, loss_png, loss_csv)
        print(f"Saved training loss CSV to {loss_csv}")
        print(f"Saved training loss plot to {loss_png}")

def main():
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = torch.cuda.current_device()
    torch.cuda.set_device(local_rank)

    train(rank, world_size)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
