# prm_prefix_examples_plot.py
# Apply a learned prefix vector to selected examples, print question & steps,
# and save per-step reward plots (original vs. prefixed) + a JSON sidecar.

import os
import json
import random
import textwrap
from typing import List, Tuple, Optional, Dict, Any

# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, DEVICE
# tensors / data
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# model / tokenizer
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork
# utils
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================
# Configurable parameters
# =========================
JSONL_PATH = "phase2_test.jsonl"
DATASET_SIZE = 500          # how many lines to make available from JSONL
OUTDIR = "prefix_per_step_plots"
PREFIX_PATH = "prefix_epochs3_batch2_nvecs1_lr0.01_size2000.pt"

# Choose specific examples by *dataset index* (0-based into the truncated DATASET_SIZE).
# If empty, we'll pick N_EXAMPLES randomly (seeded).
EXAMPLE_IDXS: List[int] = [] #[3, 97, 123, 777, 1024]   # <-- edit these indices, or leave []
N_EXAMPLES = 5
RANDOM_SEED = 4798

# Plot config
FIGSIZE = (10, 5)
BINS = 50  # (not used here, but left for easy extension)


# =========================
# Dataset
# =========================
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


# =========================
# Prefix insertion helper
# =========================
def insert_prefix(inputs, inputs_embeds: torch.Tensor, prefix: torch.Tensor):
    """
    Insert `prefix` embeddings at the first 'answer' token for each item.
    Returns new (inputs_embeds, attention_mask, answer_flag, reward_flags).
    """
    prefix_len = prefix.shape[0]
    batch_inputs_embeds = []

    for embed, af in zip(inputs_embeds, inputs.data["answer_flag"]):
        # Position of the first answer token
        index = torch.nonzero(af)[0]
        batch_inputs_embeds.append(torch.vstack((embed[:index], prefix, embed[index:])))

    prefixed_inputs_embeds = torch.stack(batch_inputs_embeds)

    # pad left by prefix_len to match new sequence length
    prefixed_attention_mask = torch.nn.functional.pad(inputs.data["attention_mask"], (prefix_len, 0))
    prefixed_answer_flag    = torch.nn.functional.pad(inputs.data["answer_flag"],    (prefix_len, 0))
    prefixed_reward_flags   = torch.nn.functional.pad(inputs.data["reward_flags"],   (prefix_len, 0))

    return prefixed_inputs_embeds, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags


# =========================
# Reward extraction helper
# =========================
def extract_step_rewards(forward, reward_flags, idx: int) -> List[float]:
    """
    Return a python list of rewards over valid steps for the idx-th item.
    """
    mask = reward_flags[idx].bool()
    rewards_tensor = forward.rewards[idx][mask]
    return rewards_tensor.detach().float().cpu().tolist()


# =========================
# Plotting
# =========================
def shorten(text: str, max_len: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def plot_side_by_side_bars(example_tag: str,
                           rewards_orig: List[float],
                           rewards_mod: List[float],
                           title: str,
                           outdir: str):
    os.makedirs(outdir, exist_ok=True)

    steps_orig = len(rewards_orig)
    steps_mod  = len(rewards_mod)
    max_steps  = max(steps_orig, steps_mod)

    arr_orig = np.full(max_steps, np.nan, dtype=float)
    arr_mod  = np.full(max_steps, np.nan, dtype=float)
    arr_orig[:steps_orig] = rewards_orig
    arr_mod[:steps_mod]   = rewards_mod

    x = np.arange(1, max_steps + 1)
    width = 0.4

    plt.figure(figsize=FIGSIZE)
    plt.bar(x - width/2, arr_orig, width=0.4, label="Original")
    plt.bar(x + width/2, arr_mod,  width=0.4, label="With prefix")
    plt.xlabel("Step number")
    plt.ylabel("PRM reward")
    plt.ylim(0, 1)  # rewards in [0, 1]
    plt.xticks(x)
    plt.title(shorten(title))
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(outdir, f"{example_tag}_per_step_rewards.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved plot: {fname}")
    return fname


# =========================
# Main logic
# =========================
def main():
    # Seeding
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Data
    ds = PRM800k(JSONL_PATH, DATASET_SIZE)
    total = len(ds)
    if total == 0:
        print("Dataset is empty; nothing to plot.")
        return

    # Choose examples
    if EXAMPLE_IDXS:
        chosen_idxs = [i for i in EXAMPLE_IDXS if 0 <= i < total]
        if len(chosen_idxs) == 0:
            print("No valid EXAMPLE_IDXS within dataset range.")
            return
    else:
        # random selection (unique)
        chosen_idxs = random.sample(range(total), k=min(N_EXAMPLES, total))

    # Model / tokenizer / prefix
    tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
    net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(DEVICE).eval()

    embedding_layer = net.pretrained_model.model.embed_tokens.weight
    prefix = torch.load(PREFIX_PATH, weights_only=True).to(DEVICE)

    os.makedirs(OUTDIR, exist_ok=True)

    # Iterate chosen examples
    for rank, idx in enumerate(chosen_idxs, start=1):
        q, steps = ds[idx]
        # make a small 1-item "original" batch
        inputs_orig = tokenizer.prepare_steps([q], [steps]).to(DEVICE)

        # Run original
        with torch.no_grad():
            fwd_orig = net(**inputs_orig, return_prob=True)
        rewards_orig = extract_step_rewards(fwd_orig, inputs_orig.data["reward_flags"], 0)

        # Build prefixed embeds at first answer step
        inputs_embeds = embedding_layer[inputs_orig.data["input_ids"]]
        prefixed_embeds, attn_mask, answer_flag, reward_flags = insert_prefix(
            inputs_orig, inputs_embeds, prefix
        )

        # Run modified (with inputs_embeds + adjusted masks)
        with torch.no_grad():
            fwd_mod = net(
                input_ids=inputs_orig.data["input_ids"],
                attention_mask=attn_mask,
                inputs_embeds=prefixed_embeds,
                return_prob=True,
            )
        rewards_mod = extract_step_rewards(fwd_mod, reward_flags, 0)

        # Console output for traceability
        print("\n" + "="*88)
        print(f"Example {rank} (dataset idx {idx})")
        print("- Question:")
        print(textwrap.fill(q, width=100))
        print("- Steps:")
        for i, s in enumerate(steps, start=1):
            print(f"  [{i:02d}] " + textwrap.fill(" ".join(s.split()), width=96, subsequent_indent="       "))

        print(f"Original steps: {len(rewards_orig)}, Prefixed steps: {len(rewards_mod)}")
        print(f"First rewards (orig): {np.round(rewards_orig[:10], 4)}")
        print(f"First rewards (mod) : {np.round(rewards_mod[:10], 4)}")

        # Plot
        tag = f"ex{rank:02d}_idx{idx}"
        plot_path = plot_side_by_side_bars(
            example_tag=tag,
            rewards_orig=rewards_orig,
            rewards_mod=rewards_mod,
            title=f"Per-step PRM rewards — ex {rank} (idx {idx})",
            outdir=OUTDIR,
        )

        # Sidecar JSON with stats
        meta: Dict[str, Any] = {
            "dataset_idx": idx,
            "question": q,
            "num_steps_text": len(steps),
            "num_steps_reward_orig": len(rewards_orig),
            "num_steps_reward_mod": len(rewards_mod),
            "mean_reward_orig": float(np.mean(rewards_orig)) if rewards_orig else None,
            "mean_reward_mod": float(np.mean(rewards_mod)) if rewards_mod else None,
            "mean_delta": (
                float(np.mean(rewards_mod) - np.mean(rewards_orig))
                if rewards_orig and rewards_mod else None
            ),
            "plot_path": plot_path,
            "prefix_path": PREFIX_PATH,
        }

        meta_path = os.path.join(OUTDIR, f"{tag}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"Wrote metadata: {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
