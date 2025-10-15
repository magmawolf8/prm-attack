# SCRIPT FOR ADVERSARIAL PREFIX OPTIMIZATION
# This script performs adversarial training to find a prefix vector that minimizes
# the reward model's score for correct answers, effectively creating an attack.
# It uses distributed data parallel (DDP) for multi-GPU training.

# --- IMPORTS ---

# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN

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

NUM_EPOCHS = 3
BATCH_SIZE = 2
NUM_PREFIX_VECTORS = 1 # The number of adversarial vectors to learn in the prefix
LEARNING_RATE = 1e-2
RANDOM_SEED = 420
DATASET_SIZE = 2000 # The number of samples to use from the .jsonl file

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
    Inserts the adversarial prefix embeddings at the start and end of the answer.
    """
    prefix_length = adversarial_prefix.shape[0]
    batch_size = batch_embeddings.shape[0]
    # This correctly gets the target device (e.g., 'cuda:5')
    device = batch_embeddings.device

    # This is created correctly on the GPU
    zeros_for_prefix = torch.zeros(prefix_length, dtype=torch.long, device=device)

    processed_embeddings_list = []
    processed_answer_flags_list = []
    processed_reward_flags_list = []

    for i in range(batch_size):
        sample_embedding = batch_embeddings[i]
        
        # <<< FIX 1: Explicitly move the flag tensors to the target GPU device >>>
        answer_flag_vector = tokenized_batch.data["answer_flag"][i].to(device)
        reward_flags_vector = tokenized_batch.data["reward_flags"][i].to(device)

        start_insertion_idx = torch.nonzero(answer_flag_vector, as_tuple=True)[0][0]
        end_insertion_idx = torch.nonzero(reward_flags_vector, as_tuple=True)[0][-1]

        new_embedding = torch.vstack((
            sample_embedding[:start_insertion_idx],
            adversarial_prefix,
            sample_embedding[start_insertion_idx:end_insertion_idx],
            adversarial_prefix,
            sample_embedding[end_insertion_idx:]
        ))
        processed_embeddings_list.append(new_embedding)

        # Now, this `torch.cat` operation will work because all tensors are on the same GPU
        new_answer_flag = torch.cat((
            answer_flag_vector[:start_insertion_idx],
            zeros_for_prefix,
            answer_flag_vector[start_insertion_idx:end_insertion_idx],
            zeros_for_prefix,
            answer_flag_vector[end_insertion_idx:]
        ))
        processed_answer_flags_list.append(new_answer_flag)

        new_reward_flag = torch.cat((
            reward_flags_vector[:start_insertion_idx],
            zeros_for_prefix,
            reward_flags_vector[start_insertion_idx:end_insertion_idx],
            zeros_for_prefix,
            reward_flags_vector[end_insertion_idx:]
        ))
        processed_reward_flags_list.append(new_reward_flag)

    prefixed_batch_embeddings = torch.stack(processed_embeddings_list)
    prefixed_answer_flag = torch.stack(processed_answer_flags_list)
    prefixed_reward_flags = torch.stack(processed_reward_flags_list)

    total_added_length = 2 * prefix_length
    prefixed_attention_mask = F.pad(
        input=tokenized_batch.data["attention_mask"],
        pad=(total_added_length, 0),
        value=1
    )

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
        save_path = f"adversarial_prefix_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_nvecs{NUM_PREFIX_VECTORS}_lr{LEARNING_RATE}_size{DATASET_SIZE}.pt"
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