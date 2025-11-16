#!/usr/bin/env python3

"""I/O and artifact saving utilities for the adversarial attack script."""

# ===============================
# imports
# ===============================

# stdlib imports
import json
import os
from datetime import datetime

# local configuration
import config as cfg

# third-party imports
import torch
import numpy as np
import matplotlib

# configure matplotlib before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===============================
# artifact functions
# ===============================

def save_hyperparams(run_dir: str):
    """Saves the high-level configuration from cfg to a JSON file."""
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


def save_metrics(loss_list, out_png, out_csv):
    """Save CSV + a panel of loss curves."""
    losses = np.array(loss_list, dtype=np.float32)  # shape: [steps, 3]

    # Save CSV with header
    header = "total_loss,nlr_loss,entropy_penalty"
    np.savetxt(out_csv, losses, delimiter=",", header=header, comments="")
    print(f"Saved training loss CSV to {out_csv}")

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
    nlr = losses[:, 1]
    H_pen = losses[:, 2]

    total_ma, total_ma_steps = moving_average(total, window=10)
    nlr_ma, nlr_ma_steps = moving_average(nlr, window=10)
    H_ma, H_ma_steps = moving_average(H_pen, window=10)

    base, ext = os.path.splitext(out_png)
    total_raw_png = base + "_total_raw" + ext
    total_ma_png = base + "_total_ma10" + ext
    nlr_raw_png = base + "_nlr_raw" + ext
    nlr_ma_png = base + "_nlr_ma10" + ext
    H_raw_png = base + "_Hpenalty_raw" + ext
    H_ma_png = base + "_Hpenalty_ma10" + ext

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

    # --- 3. NLR (raw) ---
    plt.figure(figsize=(8, 5))
    plt.plot(steps, nlr, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("NLR (avg across GPUs)")
    plt.title("Negative log reward (raw)")
    plt.tight_layout()
    plt.savefig(nlr_raw_png, dpi=150)
    plt.close()

    # --- 4. NLR (10-step moving average) ---
    plt.figure(figsize=(8, 5))
    plt.plot(nlr_ma_steps, nlr_ma, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel("NLR (10-step MA)")
    plt.title("Negative log reward (10-step moving average)")
    plt.tight_layout()
    plt.savefig(nlr_ma_png, dpi=150)
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
    
    print(f"Saved training loss plots (raw, ma10) to {os.path.dirname(out_png)}")


def save_logits(logits_tensor: torch.Tensor, out_path: str):
    """Saves a logits tensor to the specified path."""
    torch.save(logits_tensor, out_path)
    print(f"Saved logits to {out_path}")


def save_token_visualizations(probs: torch.Tensor, run_dir: str):
    """Saves grid visualizations of token probabilities for each prefix."""
    num_prefixes, vocab_size = probs.shape

    # Factorize vocabulary size into near-square dimensions
    side1 = int(np.floor(np.sqrt(vocab_size)))
    side2 = int(np.ceil(vocab_size / side1))
    print(f"Vocabulary grid size: {side1} x {side2} ({side1 * side2} >= {vocab_size})")

    vis_dir = os.path.join(run_dir, "token_prob_viz")
    os.makedirs(vis_dir, exist_ok=True)

    for i in range(num_prefixes):
        prefix_probs = probs[i]

        # Scale by maximum for visualization (avoid divide-by-zero)
        max_val = prefix_probs.max()
        if max_val > 0:
            prefix_probs = prefix_probs / max_val

        # Pad to fill the grid shape
        padded = np.zeros(side1 * side2)
        padded[:vocab_size] = prefix_probs.cpu().numpy()
        grid = padded.reshape(side1, side2)

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.title(f"Prefix {i:02d} token probabilities (scaled)")
        plt.axis("off")
        out_img = os.path.join(vis_dir, f"prefix_{i:02d}_probs_scaled.png")
        plt.savefig(out_img, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved scaled token probability visualizations to {vis_dir}")