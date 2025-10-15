# SCRIPT FOR ADVERSARIAL PREFIX OPTIMIZATION
# This script has been refactored into the ContinuousOptimizer class.

# --- IMPORTS ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

import json
import random
import time
from tqdm import tqdm
import warnings
import tempfile
import os

# --- CONFIGURATION ---
# Assuming these are in a config file or defined here
from prm_attack.config import (
    SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
)
# custom model modules
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork

warnings.filterwarnings("ignore", category=FutureWarning)

# --- DATASET DEFINITION (can be kept outside the class) ---
class PRM800k(Dataset):
    """A custom PyTorch Dataset class to load the PRM800k dataset."""
    def __init__(self, jsonl_path, size):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == size: break
                if line.strip():
                    data = json.loads(line)
                    question = data["question"]["problem"]
                    answer = data["question"]["pre_generated_steps"]
                    self.samples.append((question, answer))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# --- THE OPTIMIZER CLASS ---
class ContinuousOptimizer:
    """
    Encapsulates the logic for adversarially optimizing a continuous vector (prefix).
    """
    def __init__(self, num_epochs, batch_size,
                 learning_rate, dataset_size,
                 random_seed):
        """
        Initializes the optimizer, loading the model, tokenizer, and dataset.
        These heavy components are loaded only once.
        """
        # Store hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset_size = dataset_size
        self.random_seed = random_seed

        # Set seeds for reproducibility
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        
        print("Initializing ContinuousOptimizer...")
        # Load components that are constant across optimization runs
        self.tokenizer_api = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
        
        print("Loading reward model (this may take a moment)...")
        # Load the model to CPU first; it will be moved to GPUs in the training process
        self.reward_model = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
        self.reward_model.eval()
        # Freeze model parameters as we are only optimizing the prefix
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        print("Loading dataset...")
        # Note: Ensure 'phase2_train.jsonl' is accessible
        self.prm800k_dataset = PRM800k("phase2_train.jsonl", self.dataset_size)
        print("Initialization complete.")

    def optimize(self, initial_vector: torch.Tensor) -> torch.Tensor:
        """
        Takes an initial vector and runs the distributed optimization process.

        Args:
            initial_vector: A torch.Tensor to be optimized.

        Returns:
            The optimized torch.Tensor.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("This optimizer requires at least one GPU.")

        # Use a temporary file to get the result back from the spawned process
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            result_path = tmp.name

        try:
            # Spawn the distributed training processes
            mp.spawn(self._train_ddp,
                     args=(num_gpus, initial_vector, result_path),
                     nprocs=num_gpus,
                     join=True)

            # Load the optimized vector from the temporary file
            optimized_vector = torch.load(result_path)
        finally:
            # Clean up the temporary file
            if os.path.exists(result_path):
                os.remove(result_path)

        return optimized_vector

    def _train_ddp(self, gpu_id, num_gpus, initial_vector, result_path):
        """
        The main training function executed by each spawned DDP process.
        """
        self._setup_distributed_training(gpu_id, num_gpus)

        # --- Per-Process Setup ---
        # Move the shared model to the assigned GPU for this process
        local_reward_model = self.reward_model.to(gpu_id)
        token_embedding_layer = local_reward_model.pretrained_model.model.embed_tokens.weight
        
        # Create a local, learnable copy of the prefix on the assigned GPU
        adversarial_prefix = nn.Parameter(initial_vector.clone().to(gpu_id))
        optimizer = torch.optim.SGD([adversarial_prefix], lr=self.learning_rate)

        # Create a DDP-aware data loader
        sampler = DistributedSampler(self.prm800k_dataset, num_replicas=num_gpus, rank=gpu_id)
        data_loader = DataLoader(self.prm800k_dataset, batch_size=self.batch_size,
                                 sampler=sampler, collate_fn=self._collate_into_batch,
                                 num_workers=4, persistent_workers=True)

        if gpu_id == 0:
            print("Starting optimization...")

        # --- Training Loop ---
        for epoch in range(self.num_epochs):
            sampler.set_epoch(epoch)
            progress_bar = None
            if gpu_id == 0:
                progress_bar = tqdm(total=len(self.prm800k_dataset), desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch_questions, batch_answers in data_loader:
                tokenized_batch = self.tokenizer_api.prepare_steps(batch_questions, batch_answers)
                
                # The rest of the loop is the same as the original script
                batch_embeddings = token_embedding_layer[tokenized_batch.data["input_ids"].to(gpu_id)]
                
                p_embeds, p_mask, p_ans_flag, p_rew_flag = self._insert_adversarial_prefix(
                    tokenized_batch, batch_embeddings, adversarial_prefix
                )

                tokenized_batch.data["attention_mask"] = p_mask
                tokenized_batch.data["answer_flag"] = p_ans_flag
                tokenized_batch.data["reward_flags"] = p_rew_flag

                model_output = local_reward_model(**tokenized_batch.to(gpu_id), inputs_embeds=p_embeds, return_prob=True)
                attack_loss = -torch.log(model_output.rewards[tokenized_batch.data["reward_flags"].bool()]).mean()

                attack_loss.backward()
                dist.all_reduce(adversarial_prefix.grad, op=dist.ReduceOp.SUM)
                adversarial_prefix.grad /= num_gpus
                optimizer.step()
                optimizer.zero_grad()

                if gpu_id == 0:
                    progress_bar.update(self.batch_size * num_gpus)
                    progress_bar.set_postfix(loss=f"{attack_loss.item():.4f}")
            
            if gpu_id == 0:
                progress_bar.close()

        # --- Save Result and Cleanup ---
        dist.barrier()
        if gpu_id == 0:
            # Save the final optimized tensor to the temp file for the parent process
            torch.save(adversarial_prefix.detach().cpu(), result_path)
            print("Optimization finished.")

        self._cleanup_distributed_training()

    # --- Static Helper Methods ---
    # These functions don't depend on the class instance state (self)
    @staticmethod
    def _setup_distributed_training(gpu_id, num_gpus):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # A free port
        dist.init_process_group("nccl", rank=gpu_id, world_size=num_gpus)

    @staticmethod
    def _cleanup_distributed_training():
        dist.destroy_process_group()

    @staticmethod
    def _collate_into_batch(samples_list):
        questions, answers = zip(*samples_list)
        return list(questions), list(answers)

    @staticmethod
    def _insert_adversarial_prefix(tokenized_batch, batch_embeddings, adversarial_prefix):
        prefix_length = adversarial_prefix.shape[0]
        batch_size = batch_embeddings.shape[0]
        device = batch_embeddings.device
        zeros_for_prefix = torch.zeros(prefix_length, dtype=torch.long, device=device)
        processed_embeddings_list, processed_answer_flags_list, processed_reward_flags_list = [], [], []

        for i in range(batch_size):
            sample_embedding = batch_embeddings[i]
            answer_flag_vector = tokenized_batch.data["answer_flag"][i].to(device)
            reward_flags_vector = tokenized_batch.data["reward_flags"][i].to(device)

            start_idx = torch.nonzero(answer_flag_vector, as_tuple=True)[0][0]
            end_idx = torch.nonzero(reward_flags_vector, as_tuple=True)[0][-1]

            new_embedding = torch.vstack((
                sample_embedding[:start_idx], adversarial_prefix,
                sample_embedding[start_idx:end_idx], adversarial_prefix,
                sample_embedding[end_idx:]))
            processed_embeddings_list.append(new_embedding)

            new_ans_flag = torch.cat((
                answer_flag_vector[:start_idx], zeros_for_prefix,
                answer_flag_vector[start_idx:end_idx], zeros_for_prefix,
                answer_flag_vector[end_idx:]))
            processed_answer_flags_list.append(new_ans_flag)

            new_rew_flag = torch.cat((
                reward_flags_vector[:start_idx], zeros_for_prefix,
                reward_flags_vector[start_idx:end_idx], zeros_for_prefix,
                reward_flags_vector[end_idx:]))
            processed_reward_flags_list.append(new_rew_flag)

        p_embeds = torch.stack(processed_embeddings_list)
        p_ans_flag = torch.stack(processed_answer_flags_list)
        p_rew_flag = torch.stack(processed_reward_flags_list)
        p_mask = F.pad(input=tokenized_batch.data["attention_mask"], pad=(2 * prefix_length, 0), value=1)
        
        return p_embeds, p_mask, p_ans_flag, p_rew_flag

if __name__ == "__main__":
    optimizer = ContinuousOptimizer(num_epochs=1, batch_size=5, learning_rate=1e-2, dataset_size=2000, random_seed=420)

    num_prefix_vectors = 5
    embedding_dim = optimizer.reward_model.pretrained_model.model.embed_tokens.weight.shape[1]
    
    initial_adversarial_vector = torch.randn(num_prefix_vectors, embedding_dim)

    print(f"\nStarting optimization for a vector of shape: {initial_adversarial_vector.shape}")

    # 3. Run the optimization
    optimized_vector = optimizer.optimize(initial_adversarial_vector)

    # 4. Use the result
    print("\nOptimization complete!")
    print(f"Shape of the optimized vector: {optimized_vector.shape}")

    # Verify that the optimized vector is different from the initial one
    assert not torch.allclose(initial_adversarial_vector, optimized_vector)
    print("The optimized vector is different from the initial vector, as expected.")