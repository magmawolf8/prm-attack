#!/usr/bin/env python3
# prm_prefix_eval_multi_insert.py
# Evaluate a learned prefix tensor inserted at multiple locations:
#   (1) at the first answer token,
#   (2) before the last step/reward token,
#   (3) at both locations.
# For each strategy, compute per-trajectory Δreward = mean(reward_with_prefix) - mean(reward_original),
# and overlay the three Δ distributions on a single histogram.

import json
import random
from typing import List, Tuple, Dict

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


# --------------------------
# Config (edit as needed)
# --------------------------
JSONL_PATH = "phase2_test.jsonl"
DATASET_SIZE = 500
BATCH_SIZE = 1
NUM_WORKERS = 0
SEED = 4200
PREFIX_PATH = "adv_run_20251015_175627/aditya1_epochs2_batch2_nvecs5_lr0.01_size1500.pt"
OUT_PNG = "aditya1_delta_reward.png"
BINS = 50

# Plot appearance
ALPHA = 0.55
LABELS = {
    "first": "Insert at first answer token",
    "last":  "Insert before last reward token",
    "both":  "Insert at both locations",
}


class PRM800k(Dataset):
    def __init__(self, jsonl_path: str, size: int):
        self.samples: List[Tuple[str, List[str]]] = []
        with open(jsonl_path, "r") as f:
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


def collate_fn(batch):
    questions, answers = zip(*batch)
    return list(questions), list(answers)


def _first_answer_index(answer_flag_1d: torch.Tensor) -> int:
    """Return int index of the first token where answer_flag==1."""
    idx = torch.nonzero(answer_flag_1d, as_tuple=False)
    if idx.numel() == 0:
        # Fallback to 0 if no answer flags are set (shouldn't happen in well-formed data)
        return 0
    return int(idx[0].item())


def _last_reward_index(reward_flags_1d: torch.Tensor) -> int:
    """Return int index of the last token where reward_flags==1."""
    idx = torch.nonzero(reward_flags_1d, as_tuple=False)
    if idx.numel() == 0:
        # Fallback to the last token if no reward flags are set
        return int(reward_flags_1d.shape[0] - 1)
    return int(idx[-1].item())


def build_prefixed_inputs(
    inputs,
    inputs_embeds: torch.Tensor,
    prefix: torch.Tensor,
    strategy: str,
):
    """
    Build (inputs_embeds, attention_mask, answer_flag, reward_flags) after inserting prefix
    according to strategy in {"first", "last", "both"}.
    Assumes batch size == 1 (as in the original eval script).
    """
    assert inputs_embeds.shape[0] == 1, "This function assumes batch size 1."
    prefix_len = prefix.shape[0]

    emb = inputs_embeds[0]  # (seq_len, dim)
    af = inputs.data["answer_flag"][0]    # (seq_len,)
    rf = inputs.data["reward_flags"][0]   # (seq_len,)

    # Indices
    idx_first = _first_answer_index(af)
    idx_last  = _last_reward_index(rf)

    # Build new embedding sequence based on strategy
    if strategy == "first":
        new_emb = torch.vstack((emb[:idx_first], prefix, emb[idx_first:]))
        extra = prefix_len

        new_answer_flag = torch.nn.functional.pad(af, pad=(prefix_len, 0))
        new_reward_flag = torch.nn.functional.pad(rf, pad=(prefix_len, 0))

    elif strategy == "last":
        new_emb = torch.vstack((emb[:idx_last], prefix, emb[idx_last:]))
        extra = prefix_len

        # pad flags left by prefix_len *only once*
        new_answer_flag = torch.nn.functional.pad(af, pad=(prefix_len, 0))
        new_reward_flag = torch.nn.functional.pad(rf, pad=(prefix_len, 0))

    elif strategy == "both":
        # Insert at first, then at last (note: last is computed on ORIGINAL indices;
        # we replicate the training script behavior: insert before first answer and before last reward)
        new_emb = torch.vstack((
            emb[:idx_first],
            prefix,
            emb[idx_first:idx_last],
            prefix,
            emb[idx_last:]
        ))
        extra = 2 * prefix_len

        new_answer_flag = torch.nn.functional.pad(af, pad=(extra, 0))
        new_reward_flag = torch.nn.functional.pad(rf, pad=(extra, 0))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    new_emb = new_emb.unsqueeze(0)  # back to (1, new_seq_len, dim)
    new_attn = torch.nn.functional.pad(inputs.data["attention_mask"], pad=(extra, 0))

    return new_emb, new_attn, new_answer_flag.unsqueeze(0), new_reward_flag.unsqueeze(0)


def evaluate_and_collect_deltas_multi() -> Dict[str, List[float]]:
    # Seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Dataset / loader
    ds = PRM800k(JSONL_PATH, DATASET_SIZE)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Tokenizer / model
    tok = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
    net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(DEVICE).eval()

    # Embedding layer and prefix
    embedding_layer = net.pretrained_model.model.embed_tokens.weight
    prefix = torch.load(PREFIX_PATH, weights_only=True).to(DEVICE)

    # For each strategy: keep Δ list and running means
    strategies = ["first", "last", "both"]
    deltas: Dict[str, List[float]] = {k: [] for k in strategies}
    running = {k: {"sum_unedit": 0.0, "sum_mod": 0.0, "n": 0} for k in strategies}

    pbar = tqdm(loader, desc="Evaluating prefix effect (multi-strategy)")
    for questions, answers in pbar:
        # Original forward (no prefix)
        inputs = tok.prepare_steps(questions, answers).to(DEVICE)
        with torch.no_grad():
            out_unedit = net(**inputs, return_prob=True)
        masked_unedited = out_unedit.rewards[inputs.data["reward_flags"].bool()]
        mean_unedited = masked_unedited.mean().item()

        # Base embeddings
        inputs_embeds = embedding_layer[inputs.data["input_ids"]]

        # For each strategy, build prefixed inputs and evaluate
        for strat in strategies:
            pref_emb, attn, ans_flag, rew_flag = build_prefixed_inputs(inputs, inputs_embeds, prefix, strat)

            with torch.no_grad():
                out_mod = net(
                    input_ids=inputs.data["input_ids"],   # keep for model’s internal logic (matching original script)
                    attention_mask=attn,
                    inputs_embeds=pref_emb,
                    return_prob=True,
                )

            masked_mod = out_mod.rewards[rew_flag.bool()]
            mean_mod = masked_mod.mean().item()

            deltas[strat].append(mean_mod - mean_unedited)

            running[strat]["sum_unedit"] += mean_unedited
            running[strat]["sum_mod"] += mean_mod
            running[strat]["n"] += 1

        # Update progress bar every ~25 samples with the 'both' strategy as a representative
        r = running["both"]
        n = max(1, r["n"])
        pbar.set_postfix(
            mean_orig=f"{(r['sum_unedit']/n):.4f}",
            mean_mod=f"{(r['sum_mod']/n):.4f}",
            mean_delta=f"{((r['sum_mod']-r['sum_unedit'])/n):.4f}",
        )

    # Print summary per strategy
    print("\n=== Summary by strategy ===")
    for strat in strategies:
        r = running[strat]
        n = r["n"]
        if n == 0:
            print(f"{strat:>6}: no samples.")
            continue
        mean_orig = r["sum_unedit"] / n
        mean_mod  = r["sum_mod"]   / n
        mean_d    = (r["sum_mod"] - r["sum_unedit"]) / n
        print(f"{strat:>6}: N={n:4d} | mean_orig={mean_orig:.6f} | mean_mod={mean_mod:.6f} | mean_Δ={mean_d:.6f}")

    return deltas


def plot_overlaid_deltas(deltas_by_strategy: Dict[str, List[float]]):
    # Determine symmetric plotting range using all deltas together
    all_vals = [x for lst in deltas_by_strategy.values() for x in lst]
    if len(all_vals) == 0:
        print("No deltas computed; skipping plot.")
        return

    max_abs = max(abs(x) for x in all_vals)
    max_abs = min(1.0, max_abs * 1.05)
    if max_abs < 1e-6:
        max_abs = 0.05

    print(
        f"Max |Δreward| observed across strategies: ~{max_abs/1.05:.6f}; "
        f"plotting range [-{max_abs:.3f}, {max_abs:.3f}]"
    )

    plt.figure(figsize=(9, 5.5))
    for strat, vals in deltas_by_strategy.items():
        if len(vals) == 0:
            continue
        plt.hist(
            vals,
            bins=BINS,
            range=(-max_abs, max_abs),
            alpha=ALPHA,
            label=LABELS.get(strat, strat),
            density=False,
        )

    plt.title("Overlayed Δreward distributions by insertion strategy\n(Δ = mean(reward_with_prefix) − mean(reward_original))")
    plt.xlabel("Δreward")
    plt.ylabel("Count")
    plt.xlim(-max_abs, max_abs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved overlaid plot to {OUT_PNG}")


def main():
    # Seeds and eval
    deltas = evaluate_and_collect_deltas_multi()
    plot_overlaid_deltas(deltas)


if __name__ == "__main__":
    main()