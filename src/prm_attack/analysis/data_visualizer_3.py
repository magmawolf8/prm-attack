#!/usr/bin/env python3
import sqlite3
import ast
import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from datasets import load_dataset


class DataVisualizer:
    """
    Build and plot the Δreward distribution where:
      Δ_i = best_modified_final_reward_i  −  baseline_final_reward_i,
    with 'best_modified_final_reward_i' taken as the maximum final reward over ALL iterations
    for a given original_id (same problem), and 'baseline_final_reward_i' taken from the noop run.

    Notes:
      - Expects a table 'attacks' with columns: (commit_hash, description, original_id, mod_reward, ...)
      - mod_reward is a stringified list, e.g. "[0.12, 0.45, 0.67]"; we take the last element.
      - The modified run rows should have description like "catattack iteration {i}".
      - Optionally filters to GSM8K items with final_answer_correct=False (originally incorrect).
    """

    def __init__(
        self,
        db_path: str,
        noop_commit_hash: str,
        mod_commit_hash: str,
        filter_incorrect_only: bool = True,
        max_bins: int = 50,
        alpha: float = 0.55,
        out_png: str = "delta_reward_distribution_catattack_best.png",
        seed: int = 1234,
    ):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.noop_commit_hash = noop_commit_hash
        self.mod_commit_hash = mod_commit_hash
        self.filter_incorrect_only = filter_incorrect_only
        self.max_bins = max_bins
        self.alpha = alpha
        self.out_png = out_png
        random.seed(seed)

        # Optional filter: keep only GSM8K items that were incorrect originally
        self.gsm8k_incorrect_set: Optional[set] = None
        if filter_incorrect_only:
            self.gsm8k_incorrect_set = set()
            gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
            for entry in gsm8k:
                if not entry.get("final_answer_correct", True):
                    self.gsm8k_incorrect_set.add(entry["id"])

    # ---------- helpers ----------

    @staticmethod
    def _parse_last_reward(reward_str: str) -> Optional[float]:
        """
        Parse the last float from a string that looks like a Python list (e.g., "[0.12, 0.45]").
        Returns None on parse failure.
        """
        if not isinstance(reward_str, str):
            return None
        try:
            parsed = ast.literal_eval(reward_str)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                return float(parsed[-1])
            # Allow plain scalars too
            return float(parsed)
        except Exception:
            return None

    def _load_baseline_last_rewards(self) -> Dict[str, float]:
        """
        Map original_id -> baseline final reward from the noop run.
        Assumes description == 'noop' marks the baseline.
        If multiple rows exist for the same id, we keep the *last one fetched* (but usually there's one).
        """
        rows = self.cursor.execute(
            """
            SELECT original_id, mod_reward
            FROM attacks
            WHERE commit_hash=? AND description=?
            """,
            (self.noop_commit_hash, "noop"),
        )
        baseline: Dict[str, float] = {}
        for oid, reward_s in rows:
            last = self._parse_last_reward(reward_s)
            if last is not None:
                baseline[oid] = last
        return baseline

    def _load_best_modified_last_rewards(self) -> Dict[str, float]:
        """
        Map original_id -> best (maximum) final reward across ALL catattack iterations for the modified run.
        We select rows with description LIKE 'catattack iteration %'.
        """
        rows = self.cursor.execute(
            """
            SELECT original_id, mod_reward
            FROM attacks
            WHERE commit_hash=? AND description LIKE 'catattack iteration %'
            """,
            (self.mod_commit_hash,),
        )
        best: Dict[str, float] = {}
        for oid, reward_s in rows:
            last = self._parse_last_reward(reward_s)
            if last is None:
                continue
            if oid not in best or last > best[oid]:
                best[oid] = last
        return best

    # ---------- main API ----------

    def collect_best_deltas(self) -> List[float]:
        """
        Build the list of deltas:
          Δ = best_modified_last_reward - baseline_last_reward
        for each original_id present in BOTH runs (and passing the optional incorrect-only filter).
        """
        baseline = self._load_baseline_last_rewards()
        best_mod = self._load_best_modified_last_rewards()

        deltas: List[float] = []
        n_missing_baseline = 0
        n_missing_mod = 0
        n_filtered_out = 0

        for oid, base_last in baseline.items():
            if oid not in best_mod:
                n_missing_mod += 1
                continue
            # Optional filter: keep only originally incorrect GSM8K items
            if self.gsm8k_incorrect_set is not None and oid not in self.gsm8k_incorrect_set:
                n_filtered_out += 1
                continue

            delta = best_mod[oid] - base_last
            deltas.append(delta)

        # Some light reporting
        print(f"[DataVisualizer] Baseline items: {len(baseline)}")
        print(f"[DataVisualizer] Best-mod items: {len(best_mod)}")
        print(f"[DataVisualizer] Missing mod for baseline items: {n_missing_mod}")
        if self.gsm8k_incorrect_set is not None:
            print(f"[DataVisualizer] Filtered out (correct items): {n_filtered_out}")
        print(f"[DataVisualizer] Final matched (used) items: {len(deltas)}")

        return deltas

    def plot_best_delta_hist(self, deltas: List[float]):
        """
        Plot a single histogram of Δreward distribution (best-vs-baseline),
        with symmetric x-limits around zero (like your prefix script).
        """
        if not deltas:
            print("[DataVisualizer] No deltas to plot; aborting.")
            return

        # Symmetric range around 0, clipped
        max_abs = max(abs(x) for x in deltas)
        max_abs = min(1.0, max_abs * 1.05)
        if max_abs < 1e-6:
            max_abs = 0.05

        print(
            f"[DataVisualizer] Max |Δreward|: ~{max_abs/1.05:.6f}; "
            f"plotting range [-{max_abs:.3f}, {max_abs:.3f}]"
        )

        plt.figure(figsize=(9, 5.5))
        plt.hist(
            deltas,
            bins=self.max_bins,
            range=(-max_abs, max_abs),
            alpha=self.alpha,
            label="Best modified vs baseline",
            density=False,
        )
        plt.title("Δreward distribution (best modified over all iterations vs baseline/noop)\n"
                  "Δ = best(modified final reward) − baseline final reward")
        plt.xlabel("Δreward")
        plt.ylabel("Count")
        plt.xlim(-max_abs, max_abs)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_png, dpi=150)
        print(f"[DataVisualizer] Saved plot to {self.out_png}")

    def run(self):
        deltas = self.collect_best_deltas()
        self.plot_best_delta_hist(deltas)

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
