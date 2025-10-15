import sqlite3
from prm_attack.config import Attack, MAX_ITERATIONS
import pandas as pd
import matplotlib.pyplot as plt
import random
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple

import numpy as np

class DataVisualizer:
    # ... keep your existing __init__, _get_last_reward, _get_perf_at_catattack, visualize_mod, __del__ ...

    def __init__(self, db_path: str, noop_commit_hash: str, mod_commit_hash: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.noop_commit_hash = noop_commit_hash
        self.mod_commit_hash = mod_commit_hash

        self.gsm8k_incorrect_set = set()
        
        gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
        
        for entry in gsm8k:
            if not entry["final_answer_correct"]:
                self.gsm8k_incorrect_set.add(entry["id"])

    
    def _get_last_reward(self, reward_s):
        return float(reward_s.split()[-1].replace('[', '', -1).replace(']', '', -1))

    def _get_perf_at_catattack(self, i):
        rows = self.cursor.execute("""
SELECT a.original_id, a.mod_reward, b.mod_reward
FROM (
    SELECT original_id, mod_reward
    FROM attacks
    WHERE commit_hash=? AND description=?                                   
) as a
JOIN (
    SELECT original_id, mod_reward
    FROM attacks
    WHERE commit_hash=? AND description=?                                   
) as b
ON a.original_id=b.original_id
        """, (self.noop_commit_hash, "noop", self.mod_commit_hash, f"catattack iteration {i}"))
        
        rel_reward = list()
        for row in rows:
            if row[0] in self.gsm8k_incorrect_set:
                orig = self._get_last_reward(row[1])
                mod = self._get_last_reward(row[2])
                rel_reward.append((mod - orig))

        return rel_reward
        # join the subsets into a new table where the mod entries with that description match original_id with something in the noop table
    
    def visualize_mod(self):
        x1 = list()
        y1 = list()
        x2 = list()
        y2 = list()
        for i in range(MAX_ITERATIONS):
            rel_reward = self._get_perf_at_catattack(i)
            mean = sum(rel_reward) / len(rel_reward)
            x1.append(i)
            y1.append(mean)

            sampled_reward = random.sample(rel_reward, 100)
            x2.extend([i] * 100)
            y2.extend(sampled_reward)
            print(mean, len(rel_reward))

        plt.plot(x1, y1, label="mean reward change @ it", color="blue")
        plt.scatter(x2, y2, label="sampled reward change", color="red")

    def __del__(self):
        self.conn.close()

    # ---------- Internal helpers ----------

    def _fetch_rows(self, commit_hash: str, description: str) -> Dict[str, Tuple[str, float]]:
        """
        Return a map: original_id -> (modification_text, last_reward_float)
        for a given commit_hash and description.
        """
        rows = self.cursor.execute(
            """
            SELECT original_id, modification, mod_reward
            FROM attacks
            WHERE commit_hash=? AND description=?
            """,
            (commit_hash, description),
        ).fetchall()
        out = {}
        for oid, mod_text, reward_s in rows:
            try:
                out[oid] = (mod_text, self._get_last_reward(reward_s))
            except Exception:
                # Skip malformed reward rows
                continue
        return out

    def _build_paths(
        self,
        n_last: int = 11,
        only_incorrect: bool = True,
        require_full: bool = False,
    ) -> Tuple[
        Dict[str, Dict],               # paths per original_id
        List[Optional[float]]          # per-iteration mean Δ (ignoring missing)
    ]:
        """
        Build per-item paths of Δ reward over iterations 0..n_last (inclusive),
        where Δ = (modified_last_reward - baseline_last_reward).

        Returns:
          paths[original_id] = {
             "baseline": {"question": baseline_text, "reward": r_base},
             "iters": [
                 {"i": i, "question": mod_text_i, "reward": r_i, "delta": r_i - r_base} or None
                 for i in 0..n_last
             ]
          }
          means[i] = average Δ across available items at iteration i (None if no data).
        """
        # Baseline (noop) table
        base_map = self._fetch_rows(self.noop_commit_hash, "noop")

        # Per-iteration maps
        iter_maps: List[Dict[str, Tuple[str, float]]] = []
        for i in range(n_last + 1):
            iter_maps.append(self._fetch_rows(self.mod_commit_hash, f"catattack iteration {i}"))

        paths: Dict[str, Dict] = {}
        for oid, (base_text, r_base) in base_map.items():
            if only_incorrect and oid not in self.gsm8k_incorrect_set:
                continue

            iters: List[Optional[Dict]] = []
            full = True
            for i in range(n_last + 1):
                entry = iter_maps[i].get(oid)
                if entry is None:
                    iters.append(None)
                    full = False
                else:
                    mod_text, r_i = entry
                    iters.append({
                        "i": i,
                        "question": mod_text,
                        "reward": r_i,
                        "delta": r_i - r_base,
                    })
            if require_full and not full:
                continue

            paths[oid] = {
                "baseline": {"question": base_text, "reward": r_base},
                "iters": iters,
            }

        # Compute per-iteration means of deltas
        means: List[Optional[float]] = []
        for i in range(n_last + 1):
            vals = []
            for oid, rec in paths.items():
                node = rec["iters"][i]
                if node is not None:
                    vals.append(node["delta"])
            means.append(float(np.mean(vals)) if len(vals) > 0 else None)

        return paths, means

    def _select_examples(
        self,
        paths: Dict[str, Dict],
        n_last: int = 11,
        k: int = 3,
        successful: bool = True,
    ) -> List[str]:
        """
        Pick top/bottom-K examples based on final Δ at iteration n_last.
        Only keep items that have data at n_last.
        """
        scored: List[Tuple[str, float]] = []
        for oid, rec in paths.items():
            node = rec["iters"][n_last]
            if node is None:
                continue
            scored.append((oid, node["delta"]))
        scored.sort(key=lambda x: x[1], reverse=successful)
        return [oid for oid, _ in scored[:k]]

    # ---------- 1) Plot every path + average (thick line) ----------

    def plot_all_paths(
        self,
        n_last: int = 11,
        only_incorrect: bool = True,
        require_full: bool = False,
        max_items: Optional[int] = None,
        alpha: float = 0.15,
    ):
        """
        Draw the Δ reward trajectory for each item (thin lines),
        plus the per-iteration mean (thick line).
        """
        paths, means = self._build_paths(n_last=n_last, only_incorrect=only_incorrect, require_full=require_full)

        xs = list(range(n_last + 1))
        count = 0
        for oid, rec in paths.items():
            if max_items is not None and count >= max_items:
                break
            deltas = [node["delta"] if node is not None else np.nan for node in rec["iters"]]
            # skip if everything missing
            if all(np.isnan(deltas)):
                continue
            plt.plot(xs, deltas, color="gray", alpha=alpha, linewidth=1)
            count += 1

        # Mean line
        mean_vals = [m if m is not None else np.nan for m in means]
        plt.plot(xs, mean_vals, color="blue", linewidth=3, label="Mean Δ reward")

        plt.xlabel("CatAttack iteration (0..n)")
        plt.ylabel("Δ reward (modified − baseline)")
        plt.title(f"CatAttack Δ reward trajectories — mod commit {self.mod_commit_hash[:8]}")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()

    # ---------- 2 & 3) Print specific examples (successful / unsuccessful) ----------

    def print_examples(
        self,
        successful: bool = True,
        k: int = 3,
        n_last: int = 11,
        only_incorrect: bool = True,
        require_full: bool = False,
    ):
        """
        Print detailed per-iteration info for K examples:
          - Successful = largest positive Δ at iteration n_last
          - Unsuccessful = most negative (or smallest) Δ at iteration n_last
        Shows baseline, then each iteration's question and reward.
        """
        paths, _ = self._build_paths(n_last=n_last, only_incorrect=only_incorrect, require_full=require_full)
        picked = self._select_examples(paths, n_last=n_last, k=k, successful=successful)

        header = "SUCCESSFUL (high final Δ)" if successful else "UNSUCCESSFUL (low/negative final Δ)"
        print(f"\n=== {header}: top {len(picked)} ===\n")
        for idx, oid in enumerate(picked, 1):
            rec = paths[oid]
            base_q = rec["baseline"]["question"]
            base_r = rec["baseline"]["reward"]
            final_node = rec["iters"][n_last]
            final_delta = final_node["delta"] if final_node is not None else None

            print(f"[{idx}] original_id={oid}")
            print(f"  Baseline:")
            print(f"    reward={base_r:.6f}")
            print(f"    question={base_q}\n")

            for node in rec["iters"]:
                if node is None:
                    continue
                i = node["i"]
                r = node["reward"]
                d = node["delta"]
                q = node["question"]
                print(f"  Iter {i:02d}: reward={r:.6f}, Δ={d:+.6f}")
                print(f"    question={q}")

            print(f"\n  Final Δ at iter {n_last}: {final_delta:+.6f}" if final_delta is not None else "\n  Final Δ: (missing)")
            print("-" * 80)

    # ---------- 4) Per-example bar charts of Δ vs iteration ----------

    def plot_example_bars(
        self,
        original_id: str,
        n_last: int = 11,
        only_incorrect: bool = True,
    ):
        """
        Bar chart for Δ reward across iterations for a single item.
        """
        paths, _ = self._build_paths(n_last=n_last, only_incorrect=only_incorrect, require_full=False)
        rec = paths.get(original_id)
        if rec is None:
            raise ValueError(f"original_id {original_id} not found with current filters.")

        xs = list(range(n_last + 1))
        deltas = [node["delta"] if node is not None else 0.0 for node in rec["iters"]]
        plt.figure()
        plt.bar(xs, deltas)
        plt.xlabel("CatAttack iteration")
        plt.ylabel("Δ reward (modified − baseline)")
        plt.title(f"Δ reward by iteration — {original_id}")
        plt.grid(True, axis="y", alpha=0.2)
        plt.tight_layout()

    # ---------- Convenience: end-to-end example selection + plotting ----------

    def show_examples_with_bars(
        self,
        successful: bool = True,
        k: int = 3,
        n_last: int = 11,
        only_incorrect: bool = True,
        require_full: bool = False,
    ):
        """
        Print K examples and also render their Δ-by-iteration bar charts.
        """
        paths, _ = self._build_paths(n_last=n_last, only_incorrect=only_incorrect, require_full=require_full)
        picked = self._select_examples(paths, n_last=n_last, k=k, successful=successful)

        # Print details
        self.print_examples(successful=successful, k=k, n_last=n_last,
                            only_incorrect=only_incorrect, require_full=require_full)

        # Plot bars
        for oid in picked:
            self.plot_example_bars(oid, n_last=n_last, only_incorrect=only_incorrect)
