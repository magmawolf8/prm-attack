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
#                 Custom Datasets
#********************************

class PRM800k(Dataset):
    """
    A Dataset to load the PRM800k dataset from a local .jsonl file.
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

class SingleQADataset(Dataset):
    """
    Fake Dataset which always returns the same question and answer.
    """
    def __init__(self, question, answer_steps, size=1):
        self.question = question
        self.answer_steps = answer_steps
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # idx is ignored; always return same (question, answer)
        return self.question, self.answer_steps

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

#********************************
#                Helper Functions
#********************************

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
    questions, answers = zip(*samples_list)
    return list(questions), list(answers)

#********************************
#               Logging Utilities
#********************************

def save_loss_curve(loss_list, out_png, out_csv):
    losses = np.array(loss_list, dtype=np.float32)  # shape: [steps, 3]

    # Save CSV with header
    header = "total_loss,nll_loss,entropy_penalty"
    np.savetxt(out_csv, losses, delimiter=",", header=header, comments="")

    steps = np.arange(1, len(losses) + 1)

    # Helper: moving average with window 10 (or smaller if fewer points)
    def moving_average(x, window=10):
        window = min(window, len(x))
        if window <= 1:
            return x, steps  # nothing to smooth
        weights = np.ones(window, dtype=np.float32) / window
        ma = np.convolve(x, weights, mode="valid")
        # Align x-axis: last element of each window
        ma_steps = np.arange(window, len(x) + 1)
        return ma, ma_steps

    total = losses[:, 0]
    nll   = losses[:, 1]
    H_pen = losses[:, 2]

    total_ma, total_ma_steps = moving_average(total, window=10)
    nll_ma,   nll_ma_steps   = moving_average(nll,   window=10)
    H_ma,     H_ma_steps     = moving_average(H_pen, window=10)

    base, ext = os.path.splitext(out_png)
    total_raw_png = base + "_total_raw" + ext
    total_ma_png  = base + "_total_ma10" + ext
    nll_raw_png   = base + "_nll_raw" + ext
    nll_ma_png    = base + "_nll_ma10" + ext
    H_raw_png     = base + "_Hpenalty_raw" + ext
    H_ma_png      = base + "_Hpenalty_ma10" + ext

    # --- 1. Total loss (raw) ---
    plt.figure(figsize=(8, 5))
    plt.plot(steps, total, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("Total loss (avg across GPUs)")
    plt.title("Total attack loss (raw)")
    plt.tight_layout()
    plt.savefig(total_raw_png, dpi=150)
    plt.close()

    # --- 2. Total loss (10-step moving average) ---
    plt.figure(figsize=(8, 5))
    plt.plot(total_ma_steps, total_ma, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("Total loss (10-step MA)")
    plt.title("Total attack loss (10-step moving average)")
    plt.tight_layout()
    plt.savefig(total_ma_png, dpi=150)
    plt.close()

    # --- 3. NLL (raw) ---
    plt.figure(figsize=(8, 5))
    plt.plot(steps, nll, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("NLL (avg across GPUs)")
    plt.title("NLL loss (raw)")
    plt.tight_layout()
    plt.savefig(nll_raw_png, dpi=150)
    plt.close()

    # --- 4. NLL (10-step moving average) ---
    plt.figure(figsize=(8, 5))
    plt.plot(nll_ma_steps, nll_ma, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("NLL (10-step MA)")
    plt.title("NLL loss (10-step moving average)")
    plt.tight_layout()
    plt.savefig(nll_ma_png, dpi=150)
    plt.close()

    # --- 5. Entropy penalty (raw) ---
    plt.figure(figsize=(8, 5))
    plt.plot(steps, H_pen, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("Entropy penalty (avg across GPUs)")
    plt.title("Entropy penalty (raw)")
    plt.tight_layout()
    plt.savefig(H_raw_png, dpi=150)
    plt.close()

    # --- 6. Entropy penalty (10-step moving average) ---
    plt.figure(figsize=(8, 5))
    plt.plot(H_ma_steps, H_ma, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("Entropy penalty (10-step MA)")
    plt.title("Entropy penalty (10-step moving average)")
    plt.tight_layout()
    plt.savefig(H_ma_png, dpi=150)
    plt.close()


#********************************
#              Main training loop
#********************************

def train(gpu_id, num_gpus):
    # --- OUTPUT DIR ---
    RUN_DIR = f"adv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if gpu_id == 0:
        os.makedirs(RUN_DIR, exist_ok=True)
        print(f"Outputs will be saved under: {os.path.abspath(RUN_DIR)}")

    # --- DATA LOADING ---
    #prm800k_dataset = PRM800k("phase2_train.jsonl", DATA_SUBSET_SIZE)
    data = SingleQADataset(
        question="Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.",
        answer_steps=[
            "To determine the total weight of all Cindy's books, we need to calculate the weight of each book individually and then sum these weights.", 
            "First, for the math and science books:\n- Each math book weighs 2 pounds.\n- Each science book weighs 2 pounds.\n- Cindy has 2 math books and 2 science books.\n- Total weight of math books: \\(2 \\text{ books} \\times 2 \\text{ pounds/book} = 4 \\text{ pounds}\\).\n- Total weight of science books: \\(2 \\text{ books} \\times 2 \\text{ pounds/book} = 4 \\text{ pounds}\\).\n- Combined weight of math and science books: \\(4 \\text{ pounds} + 4 \\text{ pounds} = 8 \\text{ pounds}\\).", 
            "Second, for the French book:\n- The French book weighs 4 pounds.",
            "Third, for the English book:\n- The English book weighs 3 pounds.", 
            "Fourth, for the history book:\n- The history book weighs twice as much as the English book.\n- Weight of the history book: \\(2 \\times 3 \\text{ pounds} = 6 \\text{ pounds}\\).",
            "Finally, for the total weight:\n- Sum of the weights of all the books: \\[ 8 \\text{ pounds} \\text{ (math and science)} + 4 \\text{ pounds} \\text{ (French)} + 3 \\text{ pounds} \\text{ (English)} + 6 \\text{ pounds} \\text{ (history)} = 21 \\text{ pounds} \\]",
            "Therefore, the total weight of the books Cindy is carrying is \\(\\boxed{21}\\) pounds."
        ],
        size=BATCH_SIZE
    )

    sampler = DistributedSampler(
        data,
        num_replicas=num_gpus,
        rank=gpu_id,
        shuffle=True
    )
    data_loader = DataLoader(
        data,
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

    # SETUP
    skywork_tokenizer_api = SkyworkTokenizerAPI(
        SKYWORK_MODEL_NAME, STEP_TOKEN
    )
    reward_model = PRM_MODEL.from_pretrained(SKYWORK_MODEL_NAME).to(gpu_id).eval()
    token_embedding_layer = reward_model.pretrained_model.model.embed_tokens.weight
    vocabulary_size = token_embedding_layer.shape[0]

    adversarial_logits = torch.nn.Parameter(
        torch.randn(NUM_PREFIXES, vocabulary_size, device=gpu_id)
    )

    if gpu_id == 0:
        initial_logits_cpu = adversarial_logits.detach().cpu().clone()

    optimizer = torch.optim.Adam([adversarial_logits], lr=LEARNING_RATE, maximize=False)
    # Per-step loss history
    loss_history = []

    if gpu_id == 0:
        print("Starting training...")

    steps_per_epoch = math.ceil(len(data) / (BATCH_SIZE * num_gpus))
    total_steps = max(1, steps_per_epoch * NUM_EPOCHS)

    def get_lambda(step_idx: int) -> float:
        t = min(1.0, step_idx / total_steps)
        cos_t = 0.5 * (1 - math.cos(math.pi * t))
        return (1 - cos_t) * MIN_LAMBDA + cos_t * MAX_LAMBDA

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        if gpu_id == 0:
            progress_bar = tqdm(
                total=len(data),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
            )
        sampler.set_epoch(epoch)

        for batch_questions, batch_answers in data_loader:
            # PREPARE DATA
            tokenized_batch = skywork_tokenizer_api.prepare_steps(batch_questions, batch_answers)
            batch_embeddings = token_embedding_layer[tokenized_batch.data["input_ids"]]

            # MAKE MAGIC TOKENS
            one_hot = F.gumbel_softmax(adversarial_logits, tau=TAU, hard=False, dim=-1)
            adversarial_prefix = one_hot @ token_embedding_layer

            # INSERT MAGIC TOKENS
            prefixed_embeddings, prefixed_mask, prefixed_ans_flag, prefixed_reward_flag = insert_adversarial_prefix(
                tokenized_batch, batch_embeddings, adversarial_prefix
            )
            tokenized_batch.data["attention_mask"] = prefixed_mask
            tokenized_batch.data["answer_flag"] = prefixed_ans_flag
            tokenized_batch.data["reward_flags"] = prefixed_reward_flag
            tokenized_batch.pop("input_ids") # don't use these anymore
            tokenized_batch = tokenized_batch.to(gpu_id)

            # FORWARD PASS
            model_output = reward_model(
                **tokenized_batch,
                inputs_embeds=prefixed_embeddings,
                return_probs=True,
            )

            # CALCULATE LOSS
            nll_loss = -torch.log(
                model_output[2][tokenized_batch.data["reward_flags"].bool()]
            ).mean()

            # CALCULATE ENTROPY
            probs = torch.softmax(adversarial_logits, dim=-1)
            log_probs = (probs + 1e-12).log()
            entropy_total = -(probs * log_probs).sum()

            H_mean = entropy_total / NUM_PREFIXES
            H_norm = H_mean / math.log(vocabulary_size)
            lambda_t = get_lambda(global_step)
            H_penalty = lambda_t * H_mean

            attack_loss = nll_loss + H_penalty

            # --- METRICS FOR LOGGING ---
            with torch.no_grad():
                max_p_per_prefix = probs.max(dim=-1).values
                avg_max_p = max_p_per_prefix.mean()

                H_norm_det = H_norm.detach()

            # BACKPROPAGATION
            attack_loss.backward()

            # GRADIENT AGGREGATION AND OPTIMIZER STEP
            dist.all_reduce(adversarial_logits.grad, op=dist.ReduceOp.SUM)
            adversarial_logits.grad /= num_gpus
            adversarial_grad = adversarial_logits.grad.clone()
            optimizer.step()
            optimizer.zero_grad()

            # LOGGING
            with torch.no_grad():
                loss_tensor = attack_loss.detach()
                nll_tensor = nll_loss.detach()
                H_tensor = H_penalty.detach()

                # Average across GPUs
                for t in (loss_tensor, nll_tensor, H_tensor):
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    t /= num_gpus

                if gpu_id == 0:
                    loss_history.append((
                        float(loss_tensor.item()),
                        float(nll_tensor.item()),
                        float(H_tensor.item()),
                    ))
                    progress_bar.update(BATCH_SIZE * num_gpus)
                    progress_bar.set_postfix({
                        "loss": f"{attack_loss.item():.3f}",
                        "nll":  f"{nll_loss.item():.3f}",
                        "Hn":   f"{H_norm_det.item():.2f}",
                        "norm":   f"{torch.norm(adversarial_grad, dim=-1).mean().item():.2e}",
                        "pmax": f"{avg_max_p.item():.2f}",
                    })
            
            global_step += 1
    
        if gpu_id == 0:
            progress_bar.close()


    dist.barrier()
    # SAVE AND REPORT
    if gpu_id == 0:
        opt_path = os.path.join(
            RUN_DIR,
            f"optimized_logits.pt"
        )
        torch.save(adversarial_logits.detach().cpu(), opt_path)
        print(f"Saved optimized logits to {opt_path}")

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