#!/usr/bin/env python3
"""
catattack.py - updated with best/worst indicators and improved logging/metrics.

Key changes:
 - Adds best_so_far and worst_so_far info to the prompt context for the attacker.
 - Records per-iteration and per-item metrics and logs a concise summary after each item.
 - Sends a SUMMARY Attack record (JSON string in `modification`) to the data collector queue.
 - Prints aggregated worker-level statistics at worker shutdown.
"""

from dataclasses import dataclass, field
import ast
import queue
from typing import Any, List, Dict, Optional
from prm_attack.config import (
    SKYWORK_MODEL_NAME, ATTACKER_MODEL_NAME, DEFAULT_STEP_TOKEN,
    ATTACKER_TEMPLATE, WORLD_SIZE,
    MAX_ITERATIONS, DATA_SUBSET_LEN, RANDOM_SEED
)
from prm_attack.config import Attack
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork
from prm_attack.attacks.model_server import ModelServer
from prm_attack.analysis.data_collector import DataCollector
from datasets import load_dataset

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

from openai import BadRequestError, OpenAI

import random
import math
import argparse
import re
import json
from tqdm import tqdm
import sys
import time
import logging

class RankSummaryFilter(logging.Filter):
    """Allow logs only from rank 0 or messages containing 'SUMMARY'."""
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # keep all summaries from any rank
        if "SUMMARY" in msg:
            return True
        # otherwise only let rank 0 speak
        return self.rank == 0

# configure basic logging to stderr
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

random.seed(RANDOM_SEED)


def render_revision_history(revision_history: List[Attack], baseline_reward: float) -> str:
    """
    Format the revision history (this problem only) for the prompt.
    Each entry displays the attempted adversarial step and the relative score vs baseline.
    """
    if not revision_history:
        return "- (no previous attempts)"
    lines = []
    for r in revision_history:
        score_str = "[unknown]"
        try:
            parsed = ast.literal_eval(r.mod_reward)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                final_reward = float(parsed[-1])
                if baseline_reward != 0:
                    rel = (final_reward - baseline_reward) / baseline_reward
                    score_str = f"{rel:+.2%}"
                else:
                    score_str = f"{final_reward:+.6g}"
            else:
                final_reward = float(parsed)
                score_str = f"{final_reward:+.6g}"
        except Exception:
            score_str = "[ERROR PARSING SCORE]"
        lines.append(f"- **Previous Step:** {r.modification}\n- **Score:** {score_str}")
    return "\n".join(lines)


def build_best_block(best_rel: Optional[float], best_step: Optional[str], best_iter: Optional[int]) -> str:
    """
    Build a short summary for the best-so-far attempt to include in the prompt.
    If there is no best yet, return a placeholder indicating none.
    """
    if best_rel is None or best_step is None:
        return "- (no best attempt yet)"
    # format best_rel as signed percentage
    return f"- **Best so far (iter {best_iter}):** {best_step}\n- **Best relative change:** {best_rel:+.2%}"


def build_worst_block(worst_rel: Optional[float], worst_step: Optional[str], worst_iter: Optional[int]) -> str:
    """
    Build a short summary for the worst-so-far attempt to include in the prompt.
    """
    if worst_rel is None or worst_step is None:
        return "- (no worst attempt yet)"
    return f"- **Worst so far (iter {worst_iter}):** {worst_step}\n- **Worst relative change:** {worst_rel:+.2%}"


def build_attacker_prompt(original_question: str, current_steps: str, revision_history_block: str,
                          best_block: str, worst_block: str) -> str:
    """
    Build the attacker prompt. We pass best/worst blocks as named fields.
    The ATTACKER_TEMPLATE must include {original_question}, {current_steps}, and {revision_history_block}.
    It *may* include {best_so_far_block} and {worst_so_far_block}; if not, the extra kwargs are ignored.
    """
    return ATTACKER_TEMPLATE.format(
        original_question=original_question,
        current_steps=current_steps,
        revision_history_block=revision_history_block,
        best_so_far_block=best_block,
        worst_so_far_block=worst_block,
    )


def extract_json_object(text: str) -> dict:
    """
    Locate and parse the first JSON object found in `text`.
    Performs light cleaning before json.loads; raises json.JSONDecodeError if parsing fails.
    """
    # Escape stray backslashes that aren't part of safe JSON sequences
    text = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', text)
    # Remove stray trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    l = text.find('{')
    r = text.rfind('}')
    if l != -1 and r != -1 and r > l:
        payload = text[l:r+1]
        return json.loads(payload)
    raise json.JSONDecodeError("Could not find JSON object", text, 0)


def generate_attack(client: OpenAI, prompt: str) -> (str, str):
    """
    Query the attacker model and return tuple (raw_content, final_adversarial_step).
    We return raw_content as well so callers can log it on parse failure.
    """
    response = client.chat.completions.create(
        model=ATTACKER_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        # Consider passing temperature / top_p here as kwargs if desired
    )

    try:
        content = (response.choices[0].message.content or "")
    except Exception as e:
        raise RuntimeError(f"Unexpected attacker model response shape: {e}")

    # Try to extract JSON and the expected key
    try:
        js = extract_json_object(content)
    except json.JSONDecodeError:
        # propagate with raw content to aid debugging/logging
        raise json.JSONDecodeError("Invalid JSON from attacker", content, 0)

    key = "final_adversarial_step"
    if key in js and isinstance(js[key], str):
        return content, js[key]

    # If key missing, raise KeyError but include raw content in the message
    raise KeyError(f'Key "{key}" missing from attacker JSON. Raw content: {content}')


@torch.no_grad()
def worker_eval_gpu(rank: int, prm_q: mp.Queue, response_q: mp.Queue,
                    attacker_model_address: str, dataset, indices: list, out_q: mp.Queue):
    """
    Worker loop that:
      - computes baseline reward for each item
      - repeatedly queries attacker to propose a final appended step
      - evaluates extended steps with PRM
      - records per-iteration Attack objects and a final SUMMARY Attack
      - collects worker-level aggregated metrics and prints them at exit
    """
    filt = RankSummaryFilter(rank)
    root = logging.getLogger()
    for h in root.handlers:
        h.addFilter(filt)
    tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
    client = OpenAI(base_url=attacker_model_address, api_key="EMPTY")

    loader = DataLoader(Subset(dataset, indices), shuffle=False)
    if rank == 0:
        loader = tqdm(loader, desc="worker-0")

    # Worker-level aggregates
    worker_items = 0
    worker_best_rel_list: List[float] = []
    worker_items_with_positive_best = 0

    for entry in loader:
        # Unpack item
        try:
            item_id = entry["id"][0]
            problem = entry["problem"][0]
            steps_list = [s[0] for s in entry["steps"]]
        except Exception as e:
            logging.error("[rank %d] malformed dataset entry: %s", rank, e)
            continue

        worker_items += 1
        joined_steps = DEFAULT_STEP_TOKEN.join(steps_list)

        # Baseline evaluation
        try:
            base_inputs = tokenizer.prepare_steps(problem, steps_list)
            prm_q.put((rank, base_inputs))
            baseline_rewards = response_q.get()
            baseline_last = float(baseline_rewards[-1])
        except Exception as e:
            logging.error("[rank %d] failed to compute baseline for id=%s: %s", rank, item_id, e)
            continue

        # Per-item containers
        revision_history: List[Attack] = []
        rewards_list: List[float] = []
        rel_changes: List[float] = []
        best_rel: Optional[float] = None
        best_step: Optional[str] = None
        best_iter: Optional[int] = None
        worst_rel: Optional[float] = None
        worst_step: Optional[str] = None
        worst_iter: Optional[int] = None

        # Iterative attack loop
        for i in range(MAX_ITERATIONS):
            hist_block = render_revision_history(revision_history, baseline_last)
            best_block = build_best_block(best_rel, best_step, best_iter)
            worst_block = build_worst_block(worst_rel, worst_step, worst_iter)
            prompt = build_attacker_prompt(problem, joined_steps, hist_block, best_block, worst_block)

            try:
                raw, adversarial_step = generate_attack(client, prompt)
            except BadRequestError as e:
                logging.error("[rank %d] Attacker BadRequestError id=%s: %s", rank, item_id, e)
                break
            except json.JSONDecodeError as e:
                # Log raw content for debugging and add a failed Attack record
                logging.warning("[rank %d] Failed to parse JSON from attacker for id=%s: %s", rank, item_id, e)
                # Try to capture raw content from the exception message if available (our exception includes it)
                raw_content = getattr(e, 'doc', None) or "<raw not available>"
                logging.debug("Raw attacker output (parse failure):\n%s", raw_content)
                failed_attack = Attack(
                    original_id=item_id,
                    mod_idx=-1,
                    mod_len=1,
                    modification=f"<JSON_PARSE_ERROR:{str(e)[:200]}>",
                    mod_reward="[-1.0]",
                    description=f"iteration {i} (json parse failed)"
                )
                revision_history.append(failed_attack)
                out_q.put(failed_attack)
                # continue to next iteration (don't increment i because for loop controls index)
                continue
            except KeyError as e:
                # Log the entire raw content if included in message
                logging.warning("[rank %d] Attacker missing key for id=%s: %s", rank, item_id, e)
                raw_msg = str(e)
                logging.debug("Raw attacker output (missing key): %s", raw_msg)
                failed_attack = Attack(
                    original_id=item_id,
                    mod_idx=-1,
                    mod_len=1,
                    modification=f"<MISSING_KEY:{str(e)[:200]}>",
                    mod_reward="[-1.0]",
                    description=f"iteration {i} (missing key)"
                )
                revision_history.append(failed_attack)
                out_q.put(failed_attack)
                continue
            except Exception as e:
                logging.exception("[rank %d] Unknown attacker error for id=%s: %s", rank, item_id, e)
                failed_attack = Attack(
                    original_id=item_id,
                    mod_idx=-1,
                    mod_len=1,
                    modification=f"<ATTACKER_ERROR:{str(e)[:200]}>",
                    mod_reward="[-1.0]",
                    description=f"iteration {i} (attacker error)"
                )
                revision_history.append(failed_attack)
                out_q.put(failed_attack)
                continue

            # Evaluate extended steps with PRM
            extended_steps = steps_list + [adversarial_step]
            try:
                inputs = tokenizer.prepare_steps(problem, extended_steps)
                prm_q.put((rank, inputs))
                step_rewards = response_q.get()
                final_reward = float(step_rewards[-1])
            except Exception as e:
                logging.error("[rank %d] PRM eval failed for id=%s: %s", rank, item_id, e)
                failed_attack = Attack(
                    original_id=item_id,
                    mod_idx=-1,
                    mod_len=1,
                    modification=adversarial_step,
                    mod_reward="[-1.0]",
                    description=f"iteration {i} (prm eval failed)"
                )
                revision_history.append(failed_attack)
                out_q.put(failed_attack)
                continue

            # Compute relative change
            if baseline_last != 0:
                rel = (final_reward - baseline_last) / baseline_last
            else:
                # fallback absolute difference if baseline zero
                rel = final_reward - baseline_last

            rewards_list.append(final_reward)
            rel_changes.append(rel)

            # Update best/worst
            if (best_rel is None) or (rel > best_rel):
                best_rel = rel
                best_step = adversarial_step
                best_iter = i
            if (worst_rel is None) or (rel < worst_rel):
                worst_rel = rel
                worst_step = adversarial_step
                worst_iter = i

            # Build Attack record for this iteration and persist
            attack_record = Attack(
                original_id=item_id,
                mod_idx=i,
                mod_len=1,
                modification=adversarial_step,
                mod_reward=repr(step_rewards),
                description=f"iteration {i} (appended step)"
            )
            revision_history.append(attack_record)
            out_q.put(attack_record)

            # Logging per iteration
            logging.info("[rank %d] id=%s iter=%d rel_change=%+0.2f%%", rank, item_id, i, rel * 100.0)

        # End iterations for this item -> compute per-item summary
        item_summary: Dict[str, Any] = {
            "item_id": item_id,
            "baseline_last": baseline_last,
            "n_iterations": len(rel_changes),
            "rewards_list": rewards_list,
            "rel_changes": rel_changes,
            "best_rel": best_rel,
            "best_iter": best_iter,
            "worst_rel": worst_rel,
            "worst_iter": worst_iter
        }

        # Logging summary
        if best_rel is not None:
            logging.info("[rank %d] SUMMARY id=%s best_rel=%+0.2f%% at iter=%s n_iters=%d",
                         rank, item_id, (best_rel * 100.0), best_iter, len(rel_changes))
        else:
            logging.info("[rank %d] SUMMARY id=%s no valid attempts", rank, item_id)

        # Track worker-level aggregates
        if best_rel is not None:
            worker_best_rel_list.append(best_rel)
            if best_rel > 0:
                worker_items_with_positive_best += 1

        # Send a SUMMARY Attack object to the data collector (modification contains JSON string)
        try:
            summary_attack = Attack(
                original_id=item_id,
                mod_idx=-1,
                mod_len=len(revision_history),
                modification=json.dumps(item_summary),
                mod_reward=repr(rewards_list) if rewards_list else repr([baseline_last]),
                description="SUMMARY"
            )
            out_q.put(summary_attack)
        except Exception as e:
            logging.exception("[rank %d] Failed to queue SUMMARY for id=%s: %s", rank, item_id, e)

    # End for each item in loader: print worker-level aggregated stats
    total_items = worker_items
    pos_pct = (worker_items_with_positive_best / total_items * 100.0) if total_items > 0 else 0.0
    mean_best_rel = (sum(worker_best_rel_list) / len(worker_best_rel_list) * 100.0) if worker_best_rel_list else 0.0
    logging.info("[rank %d] Worker finished. Processed %d items. %%items_with_positive_best=%.2f%% mean_best_rel=%.2f%%",
                 rank, total_items, pos_pct, mean_best_rel)


def parallel_eval_gpu(commit_hash: str):
    """
    Orchestrate dataset sharding, start PRM server, start data collector, spawn workers.
    """
    attacker_model_addresses = [f"http://localhost:1234{i}/v1" for i in range(WORLD_SIZE)]

    # load dataset and choose subset indices
    gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
    all_indices = random.sample(list(range(len(gsm8k))), DATA_SUBSET_LEN)
    shard_size = math.ceil(DATA_SUBSET_LEN / WORLD_SIZE)
    shards = [all_indices[i:i+shard_size] for i in range(0, DATA_SUBSET_LEN, shard_size)]

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    dc = DataCollector("attacks.db", out_q, commit_hash)
    dc.start()

    # Setup PRM server (single GPU)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    model = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(device).eval()
    prm_q = ctx.Queue()
    response_qs = [ctx.Queue() for _ in range(WORLD_SIZE)]
    prm_server = ModelServer(model, device, prm_q, response_qs)
    prm_server.start()

    # Launch workers
    procs = []
    for rank, (shard, addr) in enumerate(zip(shards, attacker_model_addresses)):
        p = ctx.Process(
            target=worker_eval_gpu,
            args=(rank, prm_q, response_qs[rank], addr, gsm8k, shard, out_q),
            daemon=False
        )
        p.start()
        procs.append(p)

    # Wait for workers
    for p in procs:
        p.join()

    # Shutdown
    prm_server.stop()
    prm_server.join()

    dc.stop()
    dc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-hash", required=True)
    args = parser.parse_args()
    parallel_eval_gpu(args.commit_hash)
