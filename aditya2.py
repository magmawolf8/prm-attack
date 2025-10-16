#!/usr/bin/env python3
"""
Multi-candidate adversarial prefix optimization (DDP-capable)

- Initializes NUM_ATTACKS random prefix candidates of shape (PREFIX_LEN, embedding_dim)
- Computes pairwise cosine similarity matrix and group clustering metrics (MRL, mean angle) BEFORE training
- Optimizes each candidate independently using your reward model & DDP (all GPUs participate)
- Saves each initial and optimized prefix to SAVE_DIR
- Computes pairwise cosine similarity matrix and clustering metrics AFTER training
- Prints a short verdict if the set got tighter as a group (MRL up AND mean angle down)

Notes:
- This file assumes your environment can init NCCL process group the same way your previous script did.
- It re-uses skywork_tokenizer_api.prepare_steps(...) and insert_adversarial_prefix(...) semantics.
"""

# --- IMPORTS & CONFIG ---
import os
import random
import time
from datetime import datetime
import math
import pickle

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- user project config imports (unchanged) ---
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, NUM_EPOCHS, BATCH_SIZE, NUM_PREFIX_VECTORS, LEARNING_RATE, RANDOM_SEED, DATASET_SIZE

# model/tokenizer imports (unchanged)
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork

# --- LOCAL OVERRIDES / NEW HYPERPARAMS ---
# Interpreting NUM_PREFIX_VECTORS as prefix length (token count of the prefix).
# If your config meant "number of prefixes", set PREFIX_LEN explicitly here.
PREFIX_LEN = NUM_PREFIX_VECTORS  # length in tokens of each adversarial prefix
NUM_ATTACKS = 8                  # <- N: number of randomly-initialized prefix candidates to try (change as desired)
SAVE_DIR = f"adv_prefixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(SAVE_DIR, exist_ok=True)

# seeds
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# --- DATASET CLASS (unchanged aside from type hints) ---
import json
class PRM800k(Dataset):
    def __init__(self, jsonl_path, size):
        self.samples = []
        print(f"Loading up to {size} samples from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == size:
                    break
                if line.strip():
                    data = json.loads(line)
                    question = data["question"]["problem"]
                    answer = data["question"]["pre_generated_steps"]
                    self.samples.append((question, answer))
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --- TOKENIZER INIT (unchanged) ---
skywork_tokenizer_api = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)


# --- HELPER: insert_adversarial_prefix (copied and adapted) ---
def insert_adversarial_prefix(tokenized_batch, batch_embeddings, adversarial_prefix):
    """
    Insert the adversarial prefix (shape: [prefix_len, embed_dim]) into each sample.
    Returns:
        prefixed_batch_embeddings, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags
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
    total_added_length = 2 * prefix_length
    prefixed_attention_mask = F.pad(
        input=tokenized_batch.data["attention_mask"],
        pad=(total_added_length, 0),
        value=1
    )

    prefixed_answer_flag = torch.stack(processed_answer_flags_list)
    prefixed_reward_flags = torch.stack(processed_reward_flags_list)

    return prefixed_batch_embeddings, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags


def collate_into_batch(samples_list):
    questions, answers = zip(*samples_list)
    return list(questions), list(answers)


# --- DISTRIBUTED SETUP / TEARDOWN (keep original behavior) ---
def setup_distributed_training(gpu_id, num_gpus):
    dist.init_process_group("nccl", rank=gpu_id, world_size=num_gpus)

def cleanup_distributed_training():
    dist.destroy_process_group()


# --- VECTOR / CLUSTERING UTILITIES ---
def _flatten_prefix(prefix: torch.Tensor) -> torch.Tensor:
    return prefix.reshape(-1)

def pairwise_cosine_matrix(prefix_list):
    """
    prefix_list: list of CPU or GPU tensors shaped (prefix_len, embed_dim)
    returns: CPU tensor (N,N)
    """
    with torch.no_grad():
        vecs = []
        for p in prefix_list:
            t = p.detach().cpu()
            flat = _flatten_prefix(t)
            flat = F.normalize(flat, dim=0)
            vecs.append(flat)
        V = torch.stack(vecs, dim=0)  # (N, D)
        return V @ V.T  # (N,N) cosine similarities

def mean_resultant_length(prefix_list):
    """
    Directional clustering scalar in [0,1]. Larger -> more aligned/grouped.
    """
    with torch.no_grad():
        U = torch.stack([F.normalize(_flatten_prefix(p.detach().cpu()), dim=0) for p in prefix_list], dim=0)  # (N,D)
        centroid = U.mean(dim=0)
        return float(centroid.norm().item())

def mean_angle_to_centroid(prefix_list):
    """
    Average angular distance (radians) from each vector to the unit centroid.
    Smaller -> tighter cluster.
    """
    with torch.no_grad():
        U = torch.stack([F.normalize(_flatten_prefix(p.detach().cpu()), dim=0) for p in prefix_list], dim=0)  # (N,D)
        centroid = U.mean(dim=0)
        centroid_norm = centroid.norm().item()
        if centroid_norm == 0:
            return float(math.pi / 2.0)
        centroid_unit = centroid / centroid_norm
        cosvals = torch.clamp(U @ centroid_unit, -1.0, 1.0)
        angles = torch.acos(cosvals)
        return float(angles.mean().item())


def save_prefix_tensor(tensor, path):
    torch.save(tensor.detach().cpu(), path)

def save_matrix_csv(mat: torch.Tensor, path: str):
    import numpy as np
    np.savetxt(path, mat.numpy(), delimiter=",")


# --- TRAINING FUNCTION (DDP process entry point) ---
def train(gpu_id, num_gpus):
    setup_distributed_training(gpu_id, num_gpus)

    # --- DATA ---
    prm800k_dataset = PRM800k("phase2_train.jsonl", DATASET_SIZE)
    sampler = DistributedSampler(prm800k_dataset, num_replicas=num_gpus, rank=gpu_id, shuffle=True)
    data_loader = DataLoader(prm800k_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                             shuffle=False, num_workers=4, persistent_workers=True, collate_fn=collate_into_batch)

    if gpu_id == 0:
        print("Warming up data loader...")
        start_time = time.perf_counter()
        try:
            _ = next(iter(data_loader))
        except Exception as e:
            print("Data loader warmup failed:", e)
        end_time = time.perf_counter()
        print(f"Data loader warmup took {(end_time - start_time):.1f} seconds")

    # --- MODEL (frozen reward model) ---
    reward_model = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_model = reward_model.to(gpu_id).eval()

    token_embedding_layer = reward_model.pretrained_model.model.embed_tokens.weight
    embedding_dimension = token_embedding_layer.shape[1]

    # --- INITIALIZE PREFIX CANDIDATES ---
    # initialize on rank 0 then broadcast to all ranks (we keep tensors on device)
    initial_prefixes = []
    for i in range(NUM_ATTACKS):
        if gpu_id == 0:
            t = torch.normal(0.0, (2/embedding_dimension)**0.5, size=(PREFIX_LEN, embedding_dimension), device=gpu_id)
        else:
            t = torch.empty(PREFIX_LEN, embedding_dimension, device=gpu_id)
        # broadcast in-place (must be on same device)
        dist.broadcast(t, src=0)
        # store CPU copy on each process for later usage if desired; we will use rank 0 to save/compute matrices
        initial_prefixes.append(t.detach().cpu())

    # Rank 0 computes & saves initial cosine matrix & metrics & raw prefixes
    if gpu_id == 0:
        init_mat = pairwise_cosine_matrix(initial_prefixes)
        save_matrix_csv(init_mat, os.path.join(SAVE_DIR, "cosine_init.csv"))
        torch.save(init_mat, os.path.join(SAVE_DIR, "cosine_init.pt"))

        init_R = mean_resultant_length(initial_prefixes)
        init_mean_angle = mean_angle_to_centroid(initial_prefixes)
        with open(os.path.join(SAVE_DIR, "cluster_stats_init.txt"), "w") as fh:
            fh.write(f"MRL_R={init_R:.8f}\n")
            fh.write(f"MeanAngleRad={init_mean_angle:.8f}\n")
        for idx, p in enumerate(initial_prefixes):
            save_prefix_tensor(p, os.path.join(SAVE_DIR, f"prefix_init_{idx:03d}.pt"))

        print(f"Saved initial prefixes + cosine matrix to {SAVE_DIR}")

    dist.barrier()  # ensure init data saved before training

    # --- TRAINING: optimize each candidate sequentially ---
    optimized_prefixes = []
    for cand_idx in range(NUM_ATTACKS):
        # construct trainable param from the initial on each rank (move to the rank's device)
        init_on_device = initial_prefixes[cand_idx].to(gpu_id).clone().detach().requires_grad_(True)
        adversarial_prefix = torch.nn.Parameter(init_on_device)
        optimizer = torch.optim.SGD([adversarial_prefix], lr=LEARNING_RATE, maximize=False)

        if gpu_id == 0:
            print(f"Starting training for candidate {cand_idx+1}/{NUM_ATTACKS} ...")

        for epoch in range(NUM_EPOCHS):
            if gpu_id == 0:
                progress_bar = tqdm(total=len(prm800k_dataset), desc=f"cand {cand_idx+1} | Epoch {epoch+1}/{NUM_EPOCHS}")
            sampler.set_epoch(epoch)

            for batch_questions, batch_answers in data_loader:
                # prepare tokenized batch & embeddings
                tokenized_batch = skywork_tokenizer_api.prepare_steps(batch_questions, batch_answers)
                batch_embeddings = token_embedding_layer[tokenized_batch.data["input_ids"]]

                # insert current candidate prefix (adversarial_prefix shape: prefix_len x embed_dim)
                prefixed_embeddings, prefixed_mask, prefixed_ans_flag, prefixed_reward_flag = insert_adversarial_prefix(
                    tokenized_batch, batch_embeddings, adversarial_prefix
                )

                tokenized_batch.data["attention_mask"] = prefixed_mask
                tokenized_batch.data["answer_flag"] = prefixed_ans_flag
                tokenized_batch.data["reward_flags"] = prefixed_reward_flag
                tokenized_batch = tokenized_batch.to(gpu_id)

                # forward
                model_output = reward_model(**tokenized_batch, inputs_embeds=prefixed_embeddings, return_prob=True)

                # loss: negative log reward on the reward flags (same as original)
                attack_loss = -torch.log(model_output.rewards[tokenized_batch.data["reward_flags"].bool()]).mean()

                # backward & sync grads across processes
                attack_loss.backward()
                # reduce grads across ranks
                dist.all_reduce(adversarial_prefix.grad, op=dist.ReduceOp.SUM)
                adversarial_prefix.grad /= num_gpus

                optimizer.step()
                optimizer.zero_grad()

                if gpu_id == 0:
                    progress_bar.update(BATCH_SIZE * num_gpus)
                    progress_bar.set_postfix(loss=f"{attack_loss.item():.4f}")

            if gpu_id == 0:
                progress_bar.close()

        # end of candidate training
        dist.barrier()
        # collect final candidate (take rank 0's copy, which has synchronized updates)
        final_cpu = adversarial_prefix.detach().cpu()
        if gpu_id == 0:
            optimized_prefixes.append(final_cpu)
            save_prefix_tensor(final_cpu, os.path.join(SAVE_DIR, f"prefix_opt_{cand_idx:03d}.pt"))
            print(f"Saved optimized candidate {cand_idx:03d} to {SAVE_DIR}")

    # all candidates done
    dist.barrier()
    if gpu_id == 0:
        # compute final pairwise matrix & metrics
        fin_mat = pairwise_cosine_matrix(optimized_prefixes)
        save_matrix_csv(fin_mat, os.path.join(SAVE_DIR, "cosine_final.csv"))
        torch.save(fin_mat, os.path.join(SAVE_DIR, "cosine_final.pt"))

        fin_R = mean_resultant_length(optimized_prefixes)
        fin_mean_angle = mean_angle_to_centroid(optimized_prefixes)

        with open(os.path.join(SAVE_DIR, "cluster_stats_final.txt"), "w") as fh:
            fh.write(f"MRL_R={fin_R:.8f}\n")
            fh.write(f"MeanAngleRad={fin_mean_angle:.8f}\n")

        # print summary & verdict
        print("\n=== GROUP CLUSTERING SUMMARY ===")
        print(f"MRL_R (init) : {init_R:.6f}")
        print(f"MRL_R (final): {fin_R:.6f}")
        print(f"MeanAngleRad (init) : {init_mean_angle:.6f}")
        print(f"MeanAngleRad (final): {fin_mean_angle:.6f}")
        mean_offdiag_init = init_mat[~torch.eye(NUM_ATTACKS, dtype=bool)].mean().item()
        mean_offdiag_fin  = fin_mat[~torch.eye(NUM_ATTACKS, dtype=bool)].mean().item()
        print(f"Mean off-diagonal cosine (init)->(final): {mean_offdiag_init:.6f} -> {mean_offdiag_fin:.6f}")

        tighter_group = (fin_R > init_R) and (fin_mean_angle < init_mean_angle)
        print(f"Group got tighter? {'YES' if tighter_group else 'NO'}")
        print(f"Saved results in: {os.path.abspath(SAVE_DIR)}")

    cleanup_distributed_training()


# --- MAIN: spawn DDP processes ---
def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. This script requires at least one GPU.")
        return
    print(f"Launching training across {num_gpus} GPUs (NUM_ATTACKS={NUM_ATTACKS}, PREFIX_LEN={PREFIX_LEN})")
    mp.spawn(train, args=(num_gpus,), nprocs=num_gpus, join=True)


if __name__ == "__main__":
    main()