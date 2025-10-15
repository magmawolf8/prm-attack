# prm_prefix_eval_and_plot.py
# Evaluate a learned prefix tensor inserted at the first answer step,
# compute per-trajectory Δreward = mean(reward_with_prefix) - mean(reward_original),
# and plot the distribution as a histogram.

import json
import random
from typing import List, Tuple

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
DATASET_SIZE = 500           # how many samples to evaluate
BATCH_SIZE = 1                # keep 1 to match original behavior
NUM_WORKERS = 0               # safer with model/tokenizer
SEED = 4200
PREFIX_PATH = "prefix_epochs3_batch2_nvecs1_lr0.01_size2000.pt"
OUT_PNG = "delta_reward_distribution_prefix.png"
BINS = 50


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


def insertPrefix(inputs, inputs_embeds, prefix: torch.Tensor):
    """
    Insert `prefix` embeddings at the index of the first 'answer' token
    for each example, and return updated (inputs_embeds, attention_mask,
    answer_flag, reward_flags).
    """
    prefix_len = prefix.shape[0]

    batch_inputs_embeds = []
    for embed, af in zip(inputs_embeds, inputs.data["answer_flag"]):
        # index of first answer token
        index = torch.nonzero(af)[0]
        batch_inputs_embeds.append(torch.vstack((embed[:index], prefix, embed[index:])))

    prefixed_inputs_embeds = torch.stack(batch_inputs_embeds)
    prefixed_attention_mask = torch.nn.functional.pad(
        input=inputs.data["attention_mask"], pad=(prefix_len, 0)
    )
    prefixed_answer_flag = torch.nn.functional.pad(
        input=inputs.data["answer_flag"], pad=(prefix_len, 0)
    )
    prefixed_reward_flags = torch.nn.functional.pad(
        input=inputs.data["reward_flags"], pad=(prefix_len, 0)
    )

    return prefixed_inputs_embeds, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags


def evaluate_and_collect_deltas() -> List[float]:
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

    deltas: List[float] = []
    sum_unedited = 0.0
    sum_modified = 0.0
    n = 0

    # Iterate
    pbar = tqdm(loader, desc="Evaluating prefix effect")
    for questions, answers in pbar:
        # Batch size is 1 (to match original code logic)
        inputs = tok.prepare_steps(questions, answers)
        inputs = inputs.to(DEVICE)

        # Original forward (no prefix)
        with torch.no_grad():
            forward_unedited = net(**inputs, return_prob=True)

        masked_unedited = forward_unedited.rewards[inputs.data["reward_flags"].bool()]
        mean_unedited = masked_unedited.mean().item()

        # Build inputs_embeds with prefix at first answer step
        inputs_embeds = embedding_layer[inputs.data["input_ids"]]
        prefixed_embeds, attn_mask, answer_flag, reward_flags = insertPrefix(inputs, inputs_embeds, prefix)

        # Modified forward (with prefix inserted)
        with torch.no_grad():
            forward_modified = net(
                input_ids=inputs.data["input_ids"],
                attention_mask=attn_mask,
                inputs_embeds=prefixed_embeds,
                return_prob=True,
            )

        masked_modified = forward_modified.rewards[reward_flags.bool()]
        mean_modified = masked_modified.mean().item()

        deltas.append(mean_modified - mean_unedited)

        sum_unedited += mean_unedited
        sum_modified += mean_modified
        n += 1

        if n % 25 == 0:
            pbar.set_postfix(
                mean_orig=f"{(sum_unedited/n):.4f}",
                mean_mod=f"{(sum_modified/n):.4f}",
                mean_delta=f"{((sum_modified - sum_unedited)/n):.4f}",
            )

    print(f"\nSamples evaluated: {n}")
    if n > 0:
        print(f"Mean reward (original): {(sum_unedited/n):.6f}")
        print(f"Mean reward (with prefix): {(sum_modified/n):.6f}")
        print(f"Mean Δreward: {((sum_modified - sum_unedited)/n):.6f}")

    return deltas


def plot_deltas(deltas: List[float]):
    if len(deltas) == 0:
        print("No deltas computed; skipping plot.")
        return

    # Dynamic symmetric range based on max |Δ|
    max_abs = max(abs(x) for x in deltas)
    max_abs = min(1.0, max_abs * 1.05)
    if max_abs < 1e-6:
        max_abs = 0.05

    print(f"Max |Δreward| observed: ~{max_abs/1.05:.6f}; plotting range [-{max_abs:.3f}, {max_abs:.3f}]")

    plt.figure(figsize=(8, 5))
    plt.hist(deltas, bins=BINS, range=(-max_abs, max_abs), alpha=0.75, label="prefix (Δ)")
    plt.title("Distribution of Δreward (with prefix − original)\n(mean across valid steps)")
    plt.xlabel("Δreward")
    plt.ylabel("Count")
    plt.xlim(-max_abs, max_abs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


def main():
    deltas = evaluate_and_collect_deltas()
    plot_deltas(deltas)


if __name__ == "__main__":
    main()