"""
boutta write catattack
Prevent running if there are unstaged changes?

Given the most recent git commit's hash, launch multiple
threads (using pytorch ddp) which evaluate a certain number
of the dataset and save the resulting attacks into the database.

Try the default prompt first.
"""

from dataclasses import dataclass, field
import queue
from typing import Any
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
from prm_attack.analysis.data_reader import DataReader
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

def build_attacker_prompt(q_orig: str, steps_orig):
    """
    Builds a stateless prompt for the attacker model.
    NOTE: This version does not use any revision history.
    """
    # MODIFICATION FOR NEW TASK: Renamed 'ground_truth_answer' to 'current_steps'
    # This makes the prompt template's purpose clearer.
    return ATTACKER_TEMPLATE.format(
        original_question=q_orig,
        current_steps=steps_orig,
    )

def extract_json_object(text):
    """
    Finds and parses the first JSON object within a string.
    """
    # Clean up common formatting errors in model output
    text = re.sub(r'(?<!\\)\\(?![\\/"bfn-rtu])', r'\\\\', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)

    l, r = text.find('{'), text.rfind('}')
    if l != -1 and r != -1 and r > l:
        return json.loads(text[l:r+1])
    raise json.JSONDecodeError(msg="Could not find opening and closing braces.", doc=text, pos=0)

def generate_attack(client, prompt):
    """
    Sends a prompt to the attacker model and extracts the generated attack.
    """
    response = client.chat.completions.create(
        model=ATTACKER_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = (response.choices[0].message.content or "")
    js = extract_json_object(content)
    
    # The attacker prompt is expected to return a JSON with this key
    key = "final_adversarial_step"
    if key in js and isinstance(js[key], str):
        return js[key]
    
    raise KeyError(f'Could not find key "{key}" in the attacker model response')

@torch.no_grad()
def worker_eval_gpu(rank, prm_q, response_q, attacker_model_address, dataset, indices, q):
    """
    A worker process that generates and evaluates attacks for a subset of the dataset.

    Args:
        rank (int): The rank of this worker process.
        prm_q (Queue): The queue to send evaluation requests to the PRM model server.
        response_q (Queue): The queue to receive evaluation results from the PRM server.
        attacker_model_address (str): The URL of the attacker model server.
        dataset (Dataset): The full dataset.
        indices (list): The list of indices this worker is responsible for.
        q (Queue): The queue to send completed Attack objects to the data collector.
    """
    # Initialize the tokenizer for preparing model inputs
    tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)

    # Initialize the client to communicate with the attacker model server
    client = OpenAI(base_url=attacker_model_address, api_key="EMPTY")

    # Create a DataLoader for this worker's specific shard of the dataset
    loader = DataLoader(Subset(dataset, indices), shuffle=False)
    
    # Use a progress bar for the first worker to monitor progress
    if rank == 0:
        loader = tqdm(loader)

    # Process each data entry in the assigned shard
    for entry in loader:
        # Unpack the data from the DataLoader entry
        id = entry["id"][0]
        problem = entry["problem"][0]
        steps = [step[0] for step in entry["steps"]]

        # Generate N independent attacks for the same problem
        i = 0
        while i < MAX_ITERATIONS:
            # Build a stateless prompt for the attacker model
            attacker_prompt = build_attacker_prompt(problem, DEFAULT_STEP_TOKEN.join(steps))
            
            try:
                # Generate the new adversarial step from the attacker model
                # I've renamed `mod` to `adversarial_step` for clarity.
                adversarial_step = generate_attack(client, attacker_prompt)
            except (BadRequestError, json.JSONDecodeError, KeyError) as e:
                # If generation fails, print the error and try again
                print(f"Attacker model generation failed with error: {e}")
                continue # Skip to the next iteration without incrementing i
            
            # --- START OF KEY CHANGES ---

            # 1. Create a new list of steps with the adversarial step appended.
            extended_steps = steps + [adversarial_step]
            
            # 2. Prepare the *original problem* and the *extended steps* for the PRM model.
            #    Instead of passing `mod` as the question, we pass the original `problem`.
            inputs = tokenizer.prepare_steps(problem, extended_steps)

            # --- END OF KEY CHANGES ---

            # Send the prepared inputs to the PRM model server for evaluation
            prm_q.put((rank, inputs))
            # Wait for and retrieve the reward scores from the PRM model server
            step_rewards = response_q.get()

            # Create an Attack object to store the results
            # The `modification` now stores the adversarial step that was added.
            attack_result = Attack(
                original_id=id, 
                mod_idx=-1, # Indicates the modification is an appended step
                mod_len=1, 
                modification=adversarial_step, 
                mod_reward=repr(step_rewards), 
                description=f"catattack iteration {i} (appended step)"
            )
            
            # Put the completed attack data into the queue for the database collector
            q.put(attack_result)
            i += 1

def parallel_eval_gpu(commit_hash):
    """
    Sets up and runs the parallel evaluation across multiple worker processes.
    """
    # Define addresses for the attacker model servers
    attacker_model_addresses = [
        f"http://localhost:1234{i}/v1" for i in range(WORLD_SIZE)
    ]

    # Load the dataset and create random shards for each worker
    gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
    all_indices = random.sample(list(range(len(gsm8k))), DATA_SUBSET_LEN)
    shard_size = math.ceil(DATA_SUBSET_LEN / WORLD_SIZE)
    shards = [all_indices[i:i+shard_size] for i in range(0, DATA_SUBSET_LEN, shard_size)]

    # Set up multiprocessing context and the data collector queue
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    dc = DataCollector("attacks.db", q, commit_hash)
    dc.start()

    # Set up the primary PRM model server on the main GPU
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    model = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(device).eval()
    prm_q = ctx.Queue()
    response_qs = [ctx.Queue() for _ in range(WORLD_SIZE)]
    prm_server = ModelServer(model, device, prm_q, response_qs)
    prm_server.start()

    # Launch a worker process for each shard
    procs = list()
    for rank, shard, addr in zip(range(len(shards)), shards, attacker_model_addresses):
        p = ctx.Process(
            target=worker_eval_gpu,
            args=(rank, prm_q, response_qs[rank], addr, gsm8k, shard, q)
        )
        p.start()
        procs.append(p)

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    # Stop the PRM server and the data collector
    prm_server.stop()
    prm_server.join()

    dc.stop()
    dc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-hash", required=True)
    args = parser.parse_args()

    parallel_eval_gpu(args.commit_hash)