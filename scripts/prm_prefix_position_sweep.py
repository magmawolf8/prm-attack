# prm_prefix_position_sweep.py
# Evaluate prefix insertion at multiple positions in the answer and plot
# distributions of Δreward (with prefix − original), using mean over valid steps.

import json
import random
from typing import List, Dict, Any, Tuple

# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, DEVICE
# tensor / data modules
import torch
from torch.utils.data import Dataset, DataLoader
# models
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork
# utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


# --------------------------
# Config (edit as needed)
# --------------------------
JSONL_PATH = "phase2_test.jsonl"
DATASET_SIZE = 500          # how many samples to evaluate
BATCH_SIZE = 1               # keep 1 to match original behavior
NUM_WORKERS = 0              # safer with tokenizer/model
SEED = 4200
PREFIX_PATH = "prefix_epochs3_batch2_nvecs1_lr0.01_size2000.pt"

OUT_DIR = "prefix_position_sweep_out"
OUT_PNG = "delta_reward_distribution_positions.png"
BINS = 50


class PRM800k(Dataset):
    def __init__(self, jsonl_path: str, size: int):
        self.samples: List[Tuple[str, List[str]]] = []
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == size:
                    break
                if line.strip():
                    data = json.loads(line)
                    q = data["question"]["problem"]
                    a = data["question"]["pre_generated_steps"]
                    self.samples.append((q, a))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    questions, answers = zip(*batch)
    return list(questions), list(answers)


# --------------------------
# Prefix insertion utilities
# --------------------------
def _insert_at_start(inputs, inputs_embeds, prefix):
    """Insert just before the first answer token (uses answer_flag)."""
    prefix_len = prefix.shape[0]
    batch_inputs_embeds = []
    for embed, af in zip(inputs_embeds, inputs.data["answer_flag"]):
        index = torch.nonzero(af)[0]  # first True
        batch_inputs_embeds.append(torch.vstack((embed[:index], prefix, embed[index:])))
    stacked = torch.stack(batch_inputs_embeds)
    attn = torch.nn.functional.pad(inputs.data["attention_mask"], (prefix_len, 0))
    ansf = torch.nn.functional.pad(inputs.data["answer_flag"], (prefix_len, 0))
    rflg = torch.nn.functional.pad(inputs.data["reward_flags"], (prefix_len, 0))
    return stacked, attn, ansf, rflg


def _insert_after_first_end(inputs, inputs_embeds, prefix):
    """Insert right AFTER the end of step 1 (first reward_flag position + 1)."""
    prefix_len = prefix.shape[0]
    batch_inputs_embeds = []
    for embed, rf in zip(inputs_embeds, inputs.data["reward_flags"]):
        idx = torch.nonzero(rf)[0] + 1  # position after first end-of-step
        batch_inputs_embeds.append(torch.vstack((embed[:idx], prefix, embed[idx:])))
    stacked = torch.stack(batch_inputs_embeds)
    attn = torch.nn.functional.pad(inputs.data["attention_mask"], (prefix_len, 0))
    ansf = torch.nn.functional.pad(inputs.data["answer_flag"], (prefix_len, 0))
    rflg = torch.nn.functional.pad(inputs.data["reward_flags"], (prefix_len, 0))
    return stacked, attn, ansf, rflg


def _insert_at_mid(inputs, inputs_embeds, prefix):
    """Insert halfway through the trajectory (middle reward_flag + 1)."""
    prefix_len = prefix.shape[0]
    batch_inputs_embeds = []
    for embed, rf in zip(inputs_embeds, inputs.data["reward_flags"]):
        a = torch.nonzero(rf)
        idx = a[len(a) // 2] + 1
        batch_inputs_embeds.append(torch.vstack((embed[:idx], prefix, embed[idx:])))
    stacked = torch.stack(batch_inputs_embeds)
    attn = torch.nn.functional.pad(inputs.data["attention_mask"], (prefix_len, 0))
    ansf = torch.nn.functional.pad(inputs.data["answer_flag"], (prefix_len, 0))
    rflg = torch.nn.functional.pad(inputs.data["reward_flags"], (prefix_len, 0))
    return stacked, attn, ansf, rflg


def _insert_at_end(inputs, inputs_embeds, prefix):
    """Insert right AFTER the last reward_flag (end of trajectory)."""
    prefix_len = prefix.shape[0]
    batch_inputs_embeds = []
    for embed, rf in zip(inputs_embeds, inputs.data["reward_flags"]):
        idx = torch.nonzero(rf)[-1] + 1
        batch_inputs_embeds.append(torch.vstack((embed[:idx], prefix, embed[idx:])))
    stacked = torch.stack(batch_inputs_embeds)
    attn = torch.nn.functional.pad(inputs.data["attention_mask"], (prefix_len, 0))
    ansf = torch.nn.functional.pad(inputs.data["answer_flag"], (prefix_len, 0))
    rflg = torch.nn.functional.pad(inputs.data["reward_flags"], (prefix_len, 0))
    return stacked, attn, ansf, rflg


INSERT_FUNCS = {
    "start": _insert_at_start,
    "after_first_end": _insert_after_first_end,
    "mid": _insert_at_mid,
    "end": _insert_at_end,
}


# --------------------------
# Helpers
# --------------------------
def mean_valid_rewards(forward, reward_flags) -> float:
    masked = forward.rewards[reward_flags.bool()]
    if masked.numel() == 0:
        return float("nan")
    return masked[-1].item()


def evaluate_position_sweep() -> Dict[str, Any]:
    # Seeding
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Data
    ds = PRM800k(JSONL_PATH, DATASET_SIZE)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )

    # Tokenizer / model
    tok = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
    net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(DEVICE).eval()

    # Embedding layer and prefix
    embedding_layer = net.pretrained_model.model.embed_tokens.weight
    prefix = torch.load(PREFIX_PATH, weights_only=True).to(DEVICE)

    # Accumulators
    deltas = {name: [] for name in INSERT_FUNCS.keys()}
    sum_orig = 0.0
    n = 0

    pbar = tqdm(loader, desc="Evaluating prefix positions")
    for questions, answers in pbar:
        # Original forward
        inputs = tok.prepare_steps(questions, answers).to(DEVICE)
        with torch.no_grad():
            fwd_orig = net(**inputs, return_prob=True)
        mean_orig = mean_valid_rewards(fwd_orig, inputs.data["reward_flags"])
        if np.isnan(mean_orig):
            continue  # skip pathological example

        # Shared embeds
        inputs_embeds = embedding_layer[inputs.data["input_ids"]]

        # Try each insertion mode
        with torch.no_grad():
            for mode, fn in INSERT_FUNCS.items():
                prefixed_embeds, attn_mask, answer_flag, reward_flags = fn(
                    inputs, inputs_embeds, prefix
                )
                fwd_mod = net(
                    input_ids=inputs.data["input_ids"],
                    attention_mask=attn_mask,
                    inputs_embeds=prefixed_embeds,
                    return_prob=True,
                )
                mean_mod = mean_valid_rewards(fwd_mod, reward_flags)
                if not np.isnan(mean_mod):
                    deltas[mode].append(mean_mod - mean_orig)

        sum_orig += mean_orig
        n += 1

        if n % 25 == 0:
            pbar.set_postfix(mean_orig=f"{(sum_orig/n):.4f}")

    return dict(num_entries=n, mean_orig=sum_orig / max(n, 1), deltas=deltas)


def plot_and_save(results: Dict[str, Any]):
    os.makedirs(OUT_DIR, exist_ok=True)

    n = results["num_entries"]
    mean_orig = results["mean_orig"]
    deltas = results["deltas"]

    print(f"Samples evaluated: {n}")
    print(f"Mean reward (original): {mean_orig:.6f}")
    for k in deltas:
        if len(deltas[k]) > 0:
            print(f"Mean Δreward ({k}): {np.mean(deltas[k]):.6f} (n={len(deltas[k])})")
        else:
            print(f"Mean Δreward ({k}): N/A (n=0)")

    # Save per-position CSVs
    for k, arr in deltas.items():
        path = os.path.join(OUT_DIR, f"deltas_{k}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["delta_reward"])
            for v in arr:
                w.writerow([f"{v:.8f}"])
        print(f"Wrote {path}")

    # Combined histogram with dynamic symmetric range
    all_d = [v for arr in deltas.values() for v in arr]
    if len(all_d) == 0:
        print("No deltas computed; skipping plot.")
        return

    max_abs = max(abs(x) for x in all_d)
    max_abs = min(1.0, max_abs * 1.05)
    if max_abs < 1e-6:
        max_abs = 0.05

    print(f"Max |Δreward| observed: ~{max_abs/1.05:.6f}; plotting range [-{max_abs:.3f}, {max_abs:.3f}]")

    plt.figure(figsize=(9, 6))
    for k in ["start", "after_first_end", "mid", "end"]:
        arr = deltas[k]
        if len(arr) == 0:
            continue
        plt.hist(arr, bins=BINS, range=(-max_abs, max_abs), alpha=0.55, label=k)

    plt.title("Distribution of Δreward by prefix insertion position\n(last valid step)")
    plt.xlabel("Δreward (with prefix − original)")
    plt.ylabel("Count")
    plt.xlim(-max_abs, max_abs)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, OUT_PNG)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    results = evaluate_position_sweep()
    plot_and_save(results)


if __name__ == "__main__":
    main()