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
import pandas as pd
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


def save_metrics(metrics_series, out_png_base, out_csv):
    """
    Saves a comprehensive CSV and a 4-panel plot of training metrics.
    
    Args:
        metrics_series: A list of dictionaries, where each dict is a step's metrics.
        out_png_base: The base path for saving plots (e.g., "metrics.png").
        out_csv: The path for saving the full CSV (e.g., "metrics.csv").
    """
    if not metrics_series:
        print("No metrics to save.")
        return

    # --- 1. Convert to DataFrame and save CSV ---
    df = pd.DataFrame(metrics_series)
    df.index.name = "step"
    df.to_csv(out_csv)
    print(f"Saved full training metrics to {out_csv}")

    # --- 2. Create the 4-Panel Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Adversarial Optimization Metrics", fontsize=16, fontweight='bold')
    
    steps = df.index

    # Panel 1: Reward Progress
    ax = axes[0, 0]
    ax.plot(steps, df["soft_reward"], label="Soft Reward (Gumbel)", color="blue", alpha=0.8)
    ax.plot(steps, df["discrete_reward"], label="Discrete Reward (Hard)", color="red", linestyle="--")
    ax.set_title("Reward Progress")
    ax.set_ylabel("Reward Value")
    ax.set_xlabel("Optimization Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Loss Components
    ax = axes[0, 1]
    ax.plot(steps, df["nlr_loss"], label="NLR Loss", color="green")
    ax.plot(steps, df["H_penalty"], label="Entropy Penalty", color="purple")
    ax.set_title("Loss Components")
    ax.set_ylabel("Loss Value")
    ax.set_xlabel("Optimization Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Discreteness & Entropy
    ax = axes[1, 0]
    ax.plot(steps, df["H_norm"], label="Normalized Entropy (Hn)", color="orange")
    ax.set_ylabel("Normalized Entropy (0.0 - 1.0)")
    ax.set_ylim(0, 1.05)
    
    ax_twin = ax.twinx()
    ax_twin.plot(steps, df["avg_max_p"], label="Avg. Max Prob (p_max)", color="magenta")
    ax_twin.set_ylabel("Avg. Max Probability (0.0 - 1.0)")
    ax_twin.set_ylim(0, 1.05)

    ax.set_title("Token Discreteness")
    ax.set_xlabel("Optimization Step")
    fig.legend(loc='upper center', bbox_to_anchor=(0.7, 0.48), ncol=1)
    ax.grid(True, alpha=0.3)

    # Panel 4: Optimization Health
    ax = axes[1, 1]
    ax.plot(steps, df["grad_norm"], label="Gradient Norm", color="cyan")
    ax.set_ylabel("Gradient Norm (Log Scale)")
    ax.set_yscale("log")
    
    ax_twin = ax.twinx()
    ax_twin.plot(steps, df["lambda_t"], label="Entropy Weight (λ)", color="gray", linestyle=":")
    ax_twin.set_ylabel("Entropy Weight")

    ax.set_title("Optimization Health & Schedule")
    ax.set_xlabel("Optimization Step")
    fig.legend(loc='upper center', bbox_to_anchor=(0.3, 0.48), ncol=1)
    ax.grid(True, alpha=0.3)
    
    # Save the combined plot
    plot_path = f"{os.path.splitext(out_png_base)[0]}_metrics_panel.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved metrics panel plot to {plot_path}")


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