#!/usr/bin/env python3
"""
catattack.py

Iteratively generate adversarial final reasoning steps for problems and evaluate them
with a Process Reward Model (PRM). For each dataset example:

  1. Compute a baseline reward for the original (unmodified) steps.
  2. Repeatedly call the attacker model (which receives only the original question,
     the current steps, and the revision history for THIS problem) to propose a
     single adversarial step to append.
  3. Append that step, evaluate the extended steps with the PRM, log the result,
     and include it in the revision history for future iterations (same problem only).
"""

from dataclasses import dataclass, field
import ast
import queue
from typing import Any, List
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

random.seed(RANDOM_SEED)


def render_revision_history(revision_history: List[Attack], baseline_reward: float) -> str:
    """
    Format the revision history for the current problem into a block suitable
    for insertion into the attacker prompt.

    Each entry shows the attempted adversarial step and the relative score
    compared to the baseline_reward (signed percentage). If parsing fails, we
    include an error note for that entry.
    """
    if not revision_history:
        return "- (no previous attempts)"

    lines = []
    for r in revision_history:
        # r.mod_reward expected to be a repr(list_of_rewards) or JSON-like string
        score_str = "[unknown]"
        try:
            # Use ast.literal_eval for safety
            parsed = ast.literal_eval(r.mod_reward)
            # parsed should be a sequence (e.g., [.., final_reward])
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                final_reward = float(parsed[-1])
                if baseline_reward != 0:
                    rel = (final_reward - baseline_reward) / baseline_reward
                    score_str = f"{rel:+.2%}"
                else:
                    score_str = f"{final_reward:+.6g}"  # fall back if baseline 0
            else:
                # try parsing as single number
                final_reward = float(parsed)
                score_str = f"{final_reward:+.6g}"
        except Exception:
            score_str = "[ERROR PARSING SCORE]"

        lines.append(f"- **Previous Step:** {r.modification}\n- **Score:** {score_str}")
    return "\n".join(lines)


def build_attacker_prompt(original_question: str, current_steps: str, revision_history_block: str) -> str:
    """
    Build the attacker prompt using the final ATTACKER_TEMPLATE string.
    The template must contain {original_question}, {current_steps} and {revision_history_block}.
    """
    return ATTACKER_TEMPLATE.format(
        original_question=original_question,
        current_steps=current_steps,
        revision_history_block=revision_history_block
    )


def extract_json_object(text: str) -> dict:
    """
    Locate and parse the first JSON-like object found in `text`.
    Performs light model-output cleanup before json.loads.
    Raises json.JSONDecodeError if parsing fails.
    """
    # Escape stray backslashes that aren't part of standard JSON escape sequences
    text = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', text)
    # Remove stray trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    l = text.find('{')
    r = text.rfind('}')
    if l != -1 and r != -1 and r > l:
        payload = text[l:r+1]
        return json.loads(payload)
    raise json.JSONDecodeError("Could not find JSON object", text, 0)


def generate_attack(client: OpenAI, prompt: str) -> str:
    """
    Query the attacker model (via OpenAI client) and return the
    "final_adversarial_step" string from the JSON response.
    """
    # Send request
    response = client.chat.completions.create(
        model=ATTACKER_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        # optionally you can set temperature / max_tokens here if desired
    )

    # Extract text
    content = ""
    try:
        content = (response.choices[0].message.content or "")
    except Exception as e:
        raise RuntimeError(f"Unexpected attacker model response shape: {e}")

    js = extract_json_object(content)
    key = "final_adversarial_step"
    if key in js and isinstance(js[key], str):
        return js[key]
    raise KeyError(f'Could not find key "{key}" in attacker model response JSON')


@torch.no_grad()
def worker_eval_gpu(rank: int, prm_q: mp.Queue, response_q: mp.Queue,
                    attacker_model_address: str, dataset, indices: list, out_q: mp.Queue):
    """
    Worker process: for each assigned dataset entry, compute baseline, then iteratively
    query the attacker to propose an appended final step, evaluate it, and log.
    """
    # Tokenizer and attacker client
    tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
    client = OpenAI(base_url=attacker_model_address, api_key="EMPTY")

    # Prepare loader for this worker's shard
    loader = DataLoader(Subset(dataset, indices), shuffle=False)
    if rank == 0:
        loader = tqdm(loader, desc="worker-0")

    for entry in loader:
        try:
            item_id = entry["id"][0]
            problem = entry["problem"][0]
            steps_list = [s[0] for s in entry["steps"]]
        except Exception as e:
            print(f"[rank {rank}] malformed dataset entry: {e}", file=sys.stderr)
            continue

        # Join steps into a string for prompt; keep the token delim consistent
        joined_steps = DEFAULT_STEP_TOKEN.join(steps_list)

        # --- Step 1: compute baseline reward for the original steps ---
        try:
            base_inputs = tokenizer.prepare_steps(problem, steps_list)
            prm_q.put((rank, base_inputs))
            baseline_rewards = response_q.get()
            # baseline_rewards expected to be a sequence: use last element as final reward
            baseline_last = float(baseline_rewards[-1])
        except Exception as e:
            print(f"[rank {rank}] failed to compute baseline for id={item_id}: {e}", file=sys.stderr)
            # skip this item if we cannot get a baseline
            continue

        # Initialize per-problem revision history (list[Attack])
        revision_history: List[Attack] = []

        # Iteratively propose adversarial final steps and evaluate
        for i in range(MAX_ITERATIONS):
            # Render history block and build prompt
            hist_block = render_revision_history(revision_history, baseline_last)
            prompt = build_attacker_prompt(problem, joined_steps, hist_block)

            try:
                adversarial_step = generate_attack(client, prompt)
            except BadRequestError as e:
                print(f"[rank {rank}] Attacker model BadRequestError for id={item_id}: {e}", file=sys.stderr)
                break  # unrecoverable likely; stop iterations for this item
            except json.JSONDecodeError as e:
                # record failure in history and continue
                print(f"[rank {rank}] Failed to parse JSON from attacker for id={item_id}: {e}", file=sys.stderr)
                failed_attack = Attack(
                    original_id=item_id,
                    mod_idx=-1,
                    mod_len=1,
                    modification=f"<JSON_PARSE_ERROR:{e}>",
                    mod_reward="[-1.0]",
                    description=f"iteration {i} (json parse failed)"
                )
                revision_history.append(failed_attack)
                out_q.put(failed_attack)
                continue
            except KeyError as e:
                print(f"[rank {rank}] Attacker response missing key for id={item_id}: {e}", file=sys.stderr)
                failed_attack = Attack(
                    original_id=item_id,
                    mod_idx=-1,
                    mod_len=1,
                    modification=f"<MISSING_KEY:{e}>",
                    mod_reward="[-1.0]",
                    description=f"iteration {i} (missing key)"
                )
                revision_history.append(failed_attack)
                out_q.put(failed_attack)
                continue
            except Exception as e:
                print(f"[rank {rank}] Unknown error from attacker for id={item_id}: {e}", file=sys.stderr)
                failed_attack = Attack(
                    original_id=item_id,
                    mod_idx=-1,
                    mod_len=1,
                    modification=f"<ATTACKER_ERROR:{e}>",
                    mod_reward="[-1.0]",
                    description=f"iteration {i} (attacker error)"
                )
                revision_history.append(failed_attack)
                out_q.put(failed_attack)
                continue

            # Append adversarial step to steps and evaluate with PRM
            extended_steps = steps_list + [adversarial_step]
            try:
                inputs = tokenizer.prepare_steps(problem, extended_steps)
                prm_q.put((rank, inputs))
                step_rewards = response_q.get()
            except Exception as e:
                print(f"[rank {rank}] PRM evaluation failed for id={item_id}: {e}", file=sys.stderr)
                # record this as a failed attempt
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

            # Log attempt result
            attack_record = Attack(
                original_id=item_id,
                mod_idx=-1,
                mod_len=1,
                modification=adversarial_step,
                mod_reward=repr(step_rewards),
                description=f"iteration {i} (appended step)"
            )

            # Add to per-problem history so next iteration can use it as context
            revision_history.append(attack_record)

            # Send to central data collector for DB persistence
            out_q.put(attack_record)

            # (Optional) print progress per attempt
            if rank == 0:
                try:
                    final_reward = float(ast.literal_eval(repr(step_rewards))[-1])
                    rel = (final_reward - baseline_last) / baseline_last if baseline_last != 0 else float('inf')
                    tqdm.write(f"[rank {rank}] id={item_id} iter={i} rel_change={rel:+.2%}")
                except Exception:
                    tqdm.write(f"[rank {rank}] id={item_id} iter={i} logged result")

        # End per-item loop; proceed to next dataset entry

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