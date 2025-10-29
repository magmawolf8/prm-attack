#!/usr/bin/env python3
"""
plot_cosine_matrices.py
Load and display the cosine similarity matrices produced by the multi-candidate script:
- cosine_init.pt / cosine_init.csv
- cosine_final.pt / cosine_final.csv

It plots three panels: Initial, Final, and (Final - Initial).
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
# Comment out the next line if you want an interactive window
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_matrix(path_pt: str, path_csv: str) -> np.ndarray:
    """Try loading a torch .pt tensor first; fall back to CSV."""
    if os.path.isfile(path_pt):
        mat = torch.load(path_pt)
        if isinstance(mat, torch.Tensor):
            return mat.detach().cpu().numpy()
        # If someone saved a dict/wrapper
        return np.array(mat)
    if os.path.isfile(path_csv):
        return np.loadtxt(path_csv, delimiter=",")
    raise FileNotFoundError(f"Could not find '{path_pt}' or '{path_csv}'.")


def plot_mats(init_mat: np.ndarray, final_mat: np.ndarray, out_path: str, clamp_cos=True):
    """
    Show Initial, Final, and (Final - Initial) with consistent color scales.
    If clamp_cos=True, Initial/Final colorbar is fixed to [-1, 1] (cosine range).
    """
    # Sanity
    assert init_mat.shape == final_mat.shape, "Init and final matrices must have the same shape."
    delta = final_mat - init_mat

    # Color limits
    if clamp_cos:
        vmin = -1.0
        vmax = 1.0
    else:
        vmin = min(init_mat.min(), final_mat.min())
        vmax = max(init_mat.max(), final_mat.max())

    d_abs = np.max(np.abs(delta))
    if d_abs == 0:
        d_abs = 1e-6  # avoid degenerate colorbar
    dvmin, dvmax = -d_abs, d_abs

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    im0 = axes[0].imshow(init_mat, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Cosine (Initial)")
    axes[0].set_xlabel("Vector index")
    axes[0].set_ylabel("Vector index")
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("cosine")

    im1 = axes[1].imshow(final_mat, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Cosine (Final)")
    axes[1].set_xlabel("Vector index")
    axes[1].set_ylabel("Vector index")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("cosine")

    im2 = axes[2].imshow(delta, vmin=dvmin, vmax=dvmax, cmap="coolwarm")
    axes[2].set_title("ΔCosine = Final − Initial")
    axes[2].set_xlabel("Vector index")
    axes[2].set_ylabel("Vector index")
    cbar2 = fig.colorbar(im2, ax=axes[2])
    cbar2.set_label("delta")

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="SAVE_DIR from the training run (contains cosine_init.* / cosine_final.*)")
    ap.add_argument("--init_pt", default="cosine_init.pt")
    ap.add_argument("--init_csv", default="cosine_init.csv")
    ap.add_argument("--final_pt", default="cosine_final.pt")
    ap.add_argument("--final_csv", default="cosine_final.csv")
    ap.add_argument("--out", default="cosine_matrices.png", help="Output PNG (leave empty to show interactively)")
    ap.add_argument("--no_clamp", action="store_true", help="Do not clamp Initial/Final colormap to [-1,1]")
    args = ap.parse_args()

    init_mat = load_matrix(os.path.join(args.dir, args.init_pt),
                           os.path.join(args.dir, args.init_csv))
    final_mat = load_matrix(os.path.join(args.dir, args.final_pt),
                            os.path.join(args.dir, args.final_csv))

    plot_mats(init_mat, final_mat, os.path.join(args.dir, args.out) if args.out else "",
              clamp_cos=not args.no_clamp)


if __name__ == "__main__":
    main()