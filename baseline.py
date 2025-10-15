# SCRIPT FOR ADVERSARIAL PREFIX OPTIMIZATION
# This script performs adversarial training to find a prefix vector that minimizes
# the reward model's score for correct answers, effectively creating an attack.
# It uses distributed data parallel (DDP) for multi-GPU training.

# --- IMPORTS ---

# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, NUM_EPOCHS, BATCH_SIZE, NUM_PREFIX_VECTORS, LEARNING_RATE, RANDOM_SEED, DATASET_SIZE

# python standard libraries
import random
import time
import pickle

# tensor and deep learning modules (PyTorch)
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

# dataset modules
import json # Used for reading the .jsonl file

# custom model modules
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork

# utility modules
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# --- DATASET DEFINITION ---

class PRM800k(Dataset):
    """
    A custom PyTorch Dataset class to load the PRM800k dataset from a local .jsonl file.
    """
    def __init__(self, jsonl_path, size):
        self.samples = []
        print(f"Loading {size} samples from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                # Stop reading after reaching the desired dataset size
                if idx == size:
                    break
                if line.strip():
                    data = json.loads(line)
                    # Extract the question and the pre-generated answer steps
                    question = data["question"]["problem"]
                    answer = data["question"]["pre_generated_steps"]
                    self.samples.append((question, answer))

    def __len__(self):
        # Return the total number of samples loaded
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve a single sample by its index
        question, answer = self.samples[idx]
        return question, answer


# --- CONFIGURATION & HYPERPARAMETERS ---

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


# --- TOKENIZER INITIALIZATION ---

skywork_tokenizer_api = SkyworkTokenizerAPI(
    SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
)


# --- HELPER FUNCTIONS ---

def insert_adversarial_prefix(tokenized_batch, batch_embeddings, adversarial_prefix):
    """
    Inserts the adversarial prefix embeddings into the input embeddings.
    """
    prefix_length = adversarial_prefix.shape[0]

    processed_batch_embeddings = []
    for sample_embedding, answer_flag_vector in zip(batch_embeddings, tokenized_batch.data["answer_flag"]):
        insertion_index = torch.nonzero(answer_flag_vector)[0]
        processed_batch_embeddings.append(torch.vstack((sample_embedding[:insertion_index], adversarial_prefix, sample_embedding[insertion_index:])))

    prefixed_batch_embeddings = torch.stack(processed_batch_embeddings)
    prefixed_attention_mask = F.pad(input=tokenized_batch.data["attention_mask"], pad=(prefix_length, 0), value=1)
    prefixed_answer_flag = F.pad(input=tokenized_batch.data["answer_flag"], pad=(0, prefix_length))
    prefixed_reward_flags = F.pad(input=tokenized_batch.data["reward_flags"], pad=(prefix_length, 0))

    return prefixed_batch_embeddings, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags


def collate_into_batch(samples_list):
    """
    Custom collate function to group questions and answers from a list of samples.
    """
    questions, answers = zip(*samples_list)
    return list(questions), list(answers)


# --- DISTRIBUTED TRAINING SETUP ---

def setup_distributed_training(gpu_id, num_gpus):
    """Initialize the distributed training environment for a given process."""
    dist.init_process_group("nccl", rank=gpu_id, world_size=num_gpus)

def cleanup_distributed_training():
    """Clean up the distributed training environment."""
    dist.destroy_process_group()


# --- MAIN TRAINING LOOP ---

def train(gpu_id, num_gpus):
    """
    The main training function executed by each GPU process.
    """
    setup_distributed_training(gpu_id, num_gpus)

    # --- DATA LOADING ---
    # Load a subset of the local PRM800k dataset from the specified JSONL file.
    # Note: Make sure 'phase2_train.jsonl' is in the correct directory.
    prm800k_dataset = PRM800k("phase2_train.jsonl", DATASET_SIZE)

    sampler = DistributedSampler(prm800k_dataset, num_replicas=num_gpus, rank=gpu_id, shuffle=True)
    data_loader = DataLoader(prm800k_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=4, persistent_workers=True, collate_fn=collate_into_batch)

    if gpu_id == 0:
        print("Warming up data loader...")
        start_time = time.perf_counter()
        _ = next(iter(data_loader))
        end_time = time.perf_counter()
        print(f"Data loader warmup took {(end_time - start_time):.1f} seconds")

    # --- MODEL AND OPTIMIZER SETUP ---
    reward_model = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_model = reward_model.to(gpu_id).eval()

    token_embedding_layer = reward_model.pretrained_model.model.embed_tokens.weight
    embedding_dimension = token_embedding_layer.shape[1]
    adversarial_prefix = torch.nn.Parameter(torch.normal(0, (2/embedding_dimension)**0.5, (NUM_PREFIX_VECTORS, embedding_dimension), requires_grad=True, device=gpu_id))
    
    optimizer = torch.optim.SGD([adversarial_prefix], lr=LEARNING_RATE, maximize=False)

    if gpu_id == 0:
        print("Starting training...")

    # --- TRAINING EPOCHS ---
    for epoch in range(NUM_EPOCHS):
        if gpu_id == 0:
            progress_bar = tqdm(total=len(prm800k_dataset), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        sampler.set_epoch(epoch)

        for batch_questions, batch_answers in data_loader:
            # 1. PREPARE DATA
            tokenized_batch = skywork_tokenizer_api.prepare_steps(batch_questions, batch_answers)
            batch_embeddings = token_embedding_layer[tokenized_batch.data["input_ids"]]
            
            prefixed_embeddings, prefixed_mask, prefixed_ans_flag, prefixed_reward_flag = insert_adversarial_prefix(tokenized_batch, batch_embeddings, adversarial_prefix)
            
            tokenized_batch.data["attention_mask"] = prefixed_mask
            tokenized_batch.data["answer_flag"] = prefixed_ans_flag
            tokenized_batch.data["reward_flags"] = prefixed_reward_flag
            tokenized_batch = tokenized_batch.to(gpu_id)

            # 2. FORWARD PASS
            model_output = reward_model(**tokenized_batch, inputs_embeds=prefixed_embeddings, return_prob=True)

            # 3. CALCULATE LOSS
            attack_loss = -torch.log(model_output.rewards[tokenized_batch.data["reward_flags"].bool()]).mean()

            # 4. BACKPROPAGATION
            attack_loss.backward()

            # 5. GRADIENT AGGREGATION AND OPTIMIZER STEP
            dist.all_reduce(adversarial_prefix.grad, op=dist.ReduceOp.SUM)
            adversarial_prefix.grad /= num_gpus
            optimizer.step()
            optimizer.zero_grad()

            # 6. LOGGING
            if gpu_id == 0:
                progress_bar.update(BATCH_SIZE * num_gpus)
                progress_bar.set_postfix(loss=f"{attack_loss.item():.4f}")
        
        if gpu_id == 0:
            progress_bar.close()

    # After training, save the learned prefix. All other processes wait patiently
    dist.barrier()
    if gpu_id == 0:
        save_path = f"baseline_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_nvecs{NUM_PREFIX_VECTORS}_lr{LEARNING_RATE}_size{DATASET_SIZE}.pt"
        torch.save(adversarial_prefix, save_path)
        print(f"Saved optimized prefix to {save_path}")

    cleanup_distributed_training()


# --- SCRIPT ENTRY POINT ---

def main():
    """
    Sets up and launches the distributed training process.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. This script requires at least one GPU.")
        return
    mp.spawn(train, args=(num_gpus,), nprocs=num_gpus, join=True)


if __name__ == "__main__":
    main()