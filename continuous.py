#!/usr/bin/env python3

#********************************
#                         Imports
#********************************
# configuration
from config import *

# python standard libraries
import os
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
#         Gradient sign optimizer
#********************************

class FGSM(torch.optim.SGD):
    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr=lr, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single FGSM step."""
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

# --- CONFIGURATION & HYPERPARAMETERS ---

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
    prefixed_attention_mask = F.pad(
        input=tokenized_batch.data["attention_mask"],
        pad=(total_added_length, 0),
        value=1
    )

    return prefixed_batch_embeddings, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags


def collate_into_batch(samples_list):
    """
    Custom collate function to group questions and answers from a list of samples.
    """
    questions, answers = zip(*samples_list)
    return list(questions), list(answers)

# --- LOGGING / PLOTTING UTILS ---

def save_loss_curve(loss_list, out_png, out_csv):
    """
    Save per-step loss curve as PNG and CSV.
    """
    np.savetxt(out_csv, np.array(loss_list, dtype=np.float32), delimiter=",")
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(loss_list)+1), loss_list, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("Attack loss (avg across GPUs)")
    plt.title("Training loss curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def flat_cpu(t: torch.Tensor) -> torch.Tensor:
    """Flatten to 1D CPU tensor."""
    return t.detach().cpu().reshape(-1)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return float((a * b).sum().item())


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a - b
    return float(torch.linalg.norm(diff).item())


#********************************
#             Main training loop
#********************************

def train(gpu_id, num_gpus):
    """
    The main training function executed by each GPU process.
    """
    # --- OUTPUT DIR ---
    RUN_DIR = f"adv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if gpu_id == 0:
        os.makedirs(RUN_DIR, exist_ok=True)
        print(f"Outputs will be saved under: {os.path.abspath(RUN_DIR)}")

    # --- DATA LOADING ---
    prm800k_dataset = PRM800k("phase2_train.jsonl", DATA_SUBSET_SIZE)

    sampler = DistributedSampler(prm800k_dataset, num_replicas=num_gpus, rank=gpu_id, shuffle=True)
    data_loader = DataLoader(prm800k_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=4, persistent_workers=True, collate_fn=collate_into_batch)

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
    #adversarial_logits = torch.nn.Parameter(
    #    torch.full((NUM_PREFIXES, vocabulary_size), 1/vocabulary_size, requires_grad=True, device=gpu_id)
    #)
    adversarial_logits = torch.nn.Parameter(
        torch.full((NUM_PREFIXES, vocabulary_size), 1.0, requires_grad=True, device=gpu_id)
    )
    #adversarial_logits = torch.nn.Parameter(
    #    torch.normal(mean=1/vocabulary_size, std=1/vocabulary_size, size=(NUM_PREFIXES, vocabulary_size), requires_grad=True, device=gpu_id)
    #)
    #adversarial_logits = torch.nn.Parameter(
    #    torch.randn(NUM_PREFIXES, vocabulary_size, device=gpu_id) * 0.5
    #)


    # Save initial prefix (rank 0) for similarity / distance after training
    if gpu_id == 0:
        initial_prefix_cpu = adversarial_logits.detach().cpu().clone()

    optimizer = torch.optim.SGD([adversarial_logits], lr=LEARNING_RATE, maximize=False, momentum=MOMENTUM)

    if gpu_id == 0:
        print("Starting training...")

    # Per-step loss history (rank 0)
    loss_history = []

    # --- TRAINING EPOCHS ---
    for epoch in range(NUM_EPOCHS):
        if gpu_id == 0:
            progress_bar = tqdm(total=len(prm800k_dataset), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        sampler.set_epoch(epoch)

        for batch_questions, batch_answers in data_loader:
            # 1. PREPARE DATA
            tokenized_batch = skywork_tokenizer_api.prepare_steps(batch_questions, batch_answers)
            batch_embeddings = token_embedding_layer[tokenized_batch.data["input_ids"]]

            probs = torch.softmax(adversarial_logits, dim=-1)
            adversarial_prefix = adversarial_logits @ token_embedding_layer

            prefixed_embeddings, prefixed_mask, prefixed_ans_flag, prefixed_reward_flag = insert_adversarial_prefix(
                tokenized_batch, batch_embeddings, adversarial_prefix
            )

            tokenized_batch.data["attention_mask"] = prefixed_mask
            tokenized_batch.data["answer_flag"] = prefixed_ans_flag
            tokenized_batch.data["reward_flags"] = prefixed_reward_flag
            tokenized_batch.pop("input_ids") # don't use these anymore
            tokenized_batch = tokenized_batch.to(gpu_id)

            # 2. FORWARD PASS
            model_output = reward_model(**tokenized_batch, inputs_embeds=prefixed_embeddings, return_probs=True)
            # consider optimizing by caching keys and values for unchanged things

            # 3. CALCULATE LOSS
            nll_loss = -torch.log(model_output[2][tokenized_batch.data["reward_flags"].bool()]).mean()
            # Extra L1 loss on the logits brings many to 0
            #l1_per_prefix = torch.linalg.vector_norm(probs, ord=1, dim=-1)
            #l2_per_prefix = torch.linalg.vector_norm(probs, ord=2, dim=-1)
            mask = probs > 0
            entropy_per_prefix = -(probs[mask] * torch.log(probs[mask])).sum()
            # Could also try L2^2 loss on the probs
            #l1_penalty = L1_LAMBDA * l1_per_prefix.sum()
            #l2_2_penalty = torch.pow(l2_per_prefix, 2).sum()
            H_penalty = REG_LAMBDA * entropy_per_prefix

            #attack_loss = nll_loss + l1_penalty
            #attack_loss = l1_penalty
            #attack_loss = l2_2_penalty
            #attack_loss = nll_loss + H_penalty
            attack_loss = nll_loss

            # 4. BACKPROPAGATION
            attack_loss.backward()

            # 5. GRADIENT AGGREGATION AND OPTIMIZER STEP
            dist.all_reduce(adversarial_logits.grad, op=dist.ReduceOp.SUM)
            if gpu_id == 0:
                print("gradient vector norm:", torch.linalg.vector_norm(adversarial_logits.grad).item())
                print("Rgeularization:", H_penalty.item())
                print("first entry in probs:", probs[0][0].item())
                print("max entry in probs:", torch.max(probs[0]).item())
            adversarial_logits.grad /= num_gpus
            optimizer.step()
            optimizer.zero_grad()

            # 6. LOGGING
            with torch.no_grad():
                loss_tensor = attack_loss.detach()
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor /= num_gpus
                if gpu_id == 0:
                    loss_history.append(float(loss_tensor.item()))
                    progress_bar.update(BATCH_SIZE * num_gpus)
                    progress_bar.set_postfix({
                        "loss": f"{attack_loss.item():.4f}",
                        "Reg": f"{H_penalty.item():.4f}",
                        "near0%": f"{(adversarial_logits.abs() < 1e-7).float().mean().item() * 100:.1f}",
                    })

                    #print(torch.max(adversarial_logits))
                    #print(torch.max(probs))

        if gpu_id == 0:
            progress_bar.close()

    # --- SAVE & REPORT (rank 0) ---
    dist.barrier()
    if gpu_id == 0:
        # Save optimized prefix
        opt_path = os.path.join(
            RUN_DIR,
            f"continuous_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_nvecs{NUM_PREFIXES}_lr{LEARNING_RATE}_size{DATA_SUBSET_SIZE}.pt"
        )
        torch.save(adversarial_logits.detach().cpu(), opt_path)
        print(f"Saved optimized logits to {opt_path}")

        # Save initial prefix for reproducible comparison
        init_path = os.path.join(RUN_DIR, "initial_prefix.pt")
        torch.save(initial_prefix_cpu, init_path)

        # Compute similarity / distance on flattened tensors
        init_flat = flat_cpu(initial_prefix_cpu)
        opt_flat  = flat_cpu(adversarial_logits)

        cos_sim = cosine_similarity(init_flat, opt_flat)
        euc_dist = euclidean_distance(init_flat, opt_flat)

        # Report
        report_path = os.path.join(RUN_DIR, "prefix_change_report.txt")
        with open(report_path, "w") as f:
            f.write("=== Prefix Change Report ===\n")
            f.write(f"Initial tensor path : {init_path}\n")
            f.write(f"Optimized tensor path: {opt_path}\n")
            f.write(f"Cosine similarity   : {cos_sim:.8f}\n")
            f.write(f"Euclidean distance  : {euc_dist:.8f}\n")
        print(f"Saved report to {report_path}")
        print(f"Cosine similarity (init vs opt): {cos_sim:.6f}")
        print(f"Euclidean distance (init vs opt): {euc_dist:.6f}")

        # Save loss CSV + plot
        loss_csv = os.path.join(RUN_DIR, "training_loss.csv")
        loss_png = os.path.join(RUN_DIR, "training_loss.png")
        save_loss_curve(loss_history, loss_png, loss_csv)
        print(f"Saved training loss CSV to {loss_csv}")
        print(f"Saved training loss plot to {loss_png}")

def main():
    """
    Sets up and launches the distributed training process.
    """
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = torch.cuda.current_device()
    torch.cuda.set_device(local_rank)

    train(rank, world_size)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()