



from prm_attack.config import (
    SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, WORLD_SIZE
)
from prm_attack.config import Attack
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork
from prm_attack.analysis.data_collector import DataCollector
from datasets import load_dataset

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

import math
import argparse
from tqdm import tqdm




@torch.no_grad()
def worker_eval_gpu(rank, dataset, indices, q):
    tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)

    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    model = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(device).eval()

    loader = DataLoader(Subset(dataset, indices), shuffle=False)

    if rank == 0:
        loader = tqdm(loader)

    for entry in loader:
        id = entry["id"]
        problem = entry["problem"]
        steps = entry["steps"]
        # weird processing step I need to do to normalize problems, and steps from list[tuple[str]] into list[str]
        id = id[0]
        problem = problem[0]
        steps = [step[0] for step in steps]

        inputs = tokenizer.prepare_steps(problem, steps).to(device)

        forward = model(**inputs, return_prob=True)

        step_rewards = forward.rewards[inputs.data["reward_flags"].bool()]

        q.put(
            Attack(
                original_id=id, 
                mod_idx=-1, 
                mod_len=1, 
                modification=problem, 
                mod_reward=repr(step_rewards.tolist()), 
                description=f"noop"
            )
        )

def parallel_eval_gpu(commit_hash):
    gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
    all_indices = list(range(len(gsm8k)))
    shard_size = math.ceil(len(gsm8k) / WORLD_SIZE)
    shards = [all_indices[i:i+shard_size] for i in range(0, len(gsm8k), shard_size)]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    dc = DataCollector("attacks.db", q, commit_hash)
    dc.start()

    procs = list()
    for rank, shard in enumerate(shards):
        p = ctx.Process(
            target=worker_eval_gpu,
            args=(rank, gsm8k, shard, q)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    dc.stop()
    dc.join()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-hash", required=True)
    args = parser.parse_args()

    parallel_eval_gpu(args.commit_hash)

