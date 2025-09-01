"""boutta write catattack
Prevent running if there are unstaged changes?

Given the most recent git commit's hash, launch multiple
threads (using pytorch ddp) which evaluate a certain number
of the dataset and save the resulting attacks into the database.

Try the default prompt first.
"""




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



random.seed(RANDOM_SEED)

def render_revision_history(revision_history: list[Attack]):
    if not revision_history:
        return "- (no previous attempts)"
    lines = list()
    for r in revision_history:
        lines.append(
            f"- **Previous Question:** {r.modification}"
            f"- **Score:** {float(r.mod_reward.split()[-1].replace('[', '', -1).replace(']', '', -1))}"
        )
    return "\n".join(lines)

def build_attacker_prompt(q_orig: str, steps_orig, revision_history):
    return ATTACKER_TEMPLATE.format(
        original_question=q_orig,
        ground_truth_answer=steps_orig,
        revision_history_block=render_revision_history(revision_history)
    )

def extract_json_object(text):
    text = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)

    l, r = text.find('{'), text.rfind('}')
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(text[l:r+1])
        except Exception as e:
            print(f"could not extract json: {e}\n{text}")
            return None
    return None

def generate_attack(client, prompt):
    response = client.chat.completions.create(
        model=ATTACKER_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = (response.choices[0].message.content or "")
    js = extract_json_object(content)
    if not js:
        return None
    
    key = "final question"
    if key in js and isinstance(js[key], str):
        return js[key]
    
    return None

@torch.no_grad()
def worker_eval_gpu(rank, prm_q, response_q, attacker_model_address, dataset, indices, q):
    tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)

    client = OpenAI(base_url=attacker_model_address, api_key="EMPTY")

    loader = DataLoader(Subset(dataset, indices), shuffle=False)

    if rank == 0:
        loader = tqdm(loader)

    for entry in loader:
        id = entry["id"]
        problem = entry["problem"]
        steps = entry["steps"]
        id = id[0]
        problem = problem[0]
        steps = [step[0] for step in steps]

        revision_history = list()

        # make the attack here
        i = 0
        while i < MAX_ITERATIONS:
            attacker_prompt = build_attacker_prompt(problem, steps, revision_history)
            try:
                mod = generate_attack(client, attacker_prompt)
            except BadRequestError as e:
                print("Attacker model error: {e}")
                break
            except json.JSONDecodeError as e:
                # add failed Attack to revision_history but not to the data collector
                # decrement i (retry)
                print("Failed to parse json: {e}")
                revision_history.append(
                    Attack(
                        original_id=id, 
                        mod_idx=-1, 
                        mod_len=1, 
                        modification=f"<Failed to parse json: {e}>", 
                        mod_reward="[0.0]", 
                        description=f"catattack iteration {i}"
                    )
                )
                i -= 1
                continue

            inputs = tokenizer.prepare_steps(mod, steps)

            prm_q.put((rank, inputs))
            step_rewards = response_q.get()

            revision_history.append(
                Attack(
                    original_id=id, 
                    mod_idx=-1, 
                    mod_len=1, 
                    modification=mod, 
                    mod_reward=repr(step_rewards), 
                    description=f"catattack iteration {i}"
                )
            )

            q.put(revision_history[-1])
            i += 1

def parallel_eval_gpu(commit_hash):
    attacker_model_addresses = [
        f"http://localhost:1234{i}/v1" for i in range(WORLD_SIZE)
    ]

    gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
    all_indices = random.sample(list(range(len(gsm8k))), DATA_SUBSET_LEN)
    shard_size = math.ceil(DATA_SUBSET_LEN / WORLD_SIZE)
    shards = [all_indices[i:i+shard_size] for i in range(0, DATA_SUBSET_LEN, shard_size)]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    dc = DataCollector("attacks.db", q, commit_hash)
    dc.start()

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    model = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(device).eval()
    prm_q = ctx.Queue()
    response_qs = [ctx.Queue() for _ in range(WORLD_SIZE)]
    prm_server = ModelServer(model, device, prm_q, response_qs)
    prm_server.start()

    procs = list()
    for rank, shard, addr in zip(range(len(shards)), shards, attacker_model_addresses):
        p = ctx.Process(
            target=worker_eval_gpu,
            args=(rank, prm_q, response_qs[rank], addr, gsm8k, shard, q)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    prm_server.stop()
    prm_server.join()

    dc.stop()
    dc.join()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-hash", required=True)
    args = parser.parse_args()

    parallel_eval_gpu(args.commit_hash)

