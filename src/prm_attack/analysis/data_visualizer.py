""""""




import sqlite3
from prm_attack.config import Attack, MAX_ITERATIONS
import pandas as pd
import matplotlib.pyplot as plt
import random
from datasets import load_dataset




class DataVisualizer:
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
