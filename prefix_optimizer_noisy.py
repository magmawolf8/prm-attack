#!/usr/bin/env python3

"""Adversarial prefix optimization against Skywork PRM.

This script optimizes a soft token prefix (via Gumbel-Softmax over the
Skywork vocabulary) to maximally increase the PRM reward on a given
question/solution pair or a subset of PRM800k.
"""

# ===============================
# imports
# ===============================

# stdlib imports
import json
import math
import os
import random
import time
from datetime import datetime

# local configuration
import config as cfg
import artifacts
from prefix_optimizer import PrefixOptimizer

# third-party imports
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

# custom model modules
from skywork_tokenizer import SkyworkTokenizer
from skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL

# misc
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===============================
# custom datasets
# ===============================

class PRM800k(Dataset):
    """Dataset which serves the PRM800k dataset from a local .jsonl file."""

    def __init__(self, jsonl_path, size):
        self.samples = []
        print(f"Loading {size} samples from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            for idx, line in enumerate(f):
                if idx == size:
                    break
                if line.strip():
                    data = json.loads(line)
                    question = data["question"]["problem"]
                    answer = data["question"]["pre_generated_steps"]
                    self.samples.append((question, answer))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, answer = self.samples[idx]
        return question, answer


class SingleQADataset(Dataset):
    """Dataset which always returns the same (question, answer) pair."""

    def __init__(self, question, answer_steps, size=1):
        self.question = question
        self.answer_steps = answer_steps
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.question, self.answer_steps


# ===============================
# gradient sign optimizer
# ===============================

class FGSM(torch.optim.SGD):
    """Simple optimizer that steps in the sign of the gradient."""

    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr=lr, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad.sign(), alpha=-lr)

        return loss


# ===============================
# global seeds
# ===============================

torch.manual_seed(cfg.RANDOM_SEED)
random.seed(cfg.RANDOM_SEED)
torch.cuda.manual_seed_all(cfg.RANDOM_SEED)


# ===============================
# helper functions
# ===============================

def collate_into_batch(samples_list):
    questions, answers = zip(*samples_list)
    return list(questions), list(answers)


# ===============================
# training class
# ===============================

class NoisyPrefixOptimizer(PrefixOptimizer):
    """
    Extends PrefixOptimizer to add logging, artifact saving,
    progress bars, and discrete reward calculation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Logging-specific state
        self.noisy = (self.gpu_id == 0)
        self.run_dir = ""
        self.initial_logits_cpu = None
        self.metrics_series = []
        self.progress_bar = None

        # Setup
        self._setup_output_dir()
        
        # Save initial logits (which are created in base _setup_model_and_opt)
        if self.noisy:
            self.initial_logits_cpu = self.adversarial_logits.detach().cpu().clone()
    
    # --- Overridden Setup Methods ---

    def _setup_output_dir(self):
        """Creates the unique output directory for this run."""
        if self.noisy:
            self.run_dir = f"adv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.run_dir, exist_ok=True)
            print(f"Outputs will be saved under: {os.path.abspath(self.run_dir)}")
 
    # --- Overridden Hooks ---

    def on_train_start(self):
        if self.noisy:
            print("Starting training...")
            if self.hparams.FULL_BATCH:
                self.progress_bar = tqdm(
                    total=self.hparams.NUM_EPOCHS,
                    desc="Full-batch optimization",
                )

    def on_epoch_start(self, epoch: int):
        if self.noisy and not self.hparams.FULL_BATCH:
            self.progress_bar = tqdm(
                total=len(self.data),
                desc=f"Epoch {epoch + 1}/{self.hparams.NUM_EPOCHS}",
            )

    def on_epoch_end(self, epoch: int):
        if self.noisy:
            if self.hparams.FULL_BATCH:
                self.progress_bar.update(1)
            else:
                self.progress_bar.close()

    def on_train_end(self, final_logits_cpu: torch.Tensor):
        """Called after dist.barrier()"""
        if self.noisy:
            if self.hparams.FULL_BATCH and self.progress_bar is not None:
                self.progress_bar.close()
            
            self._save_artifacts(final_logits_cpu)

    def run_step(self, batch):
        """Runs one optimization step."""
        batch_questions, batch_answers = batch

        # PREPARE DATA
        tokenized_batch = self.skywork_tokenizer.prepare_steps(
            batch_questions, batch_answers
        )
        batch_embeddings = self.token_embedding_layer[tokenized_batch.data["input_ids"]]
        tokenized_batch.pop("input_ids")
        batch_attention_mask = tokenized_batch.data["attention_mask"]
        batch_answer_flags = tokenized_batch.data["answer_flag"]
        batch_reward_flags = tokenized_batch.data["reward_flags"]

        # MAKE MAGIC TOKENS
        adversarial_prefix = self._make_adversarial_prefix(hard=False)

        # INSERT MAGIC TOKENS
        (
            prefixed_embeds,
            prefixed_mask,
            prefixed_ans_flag,
            prefixed_reward_flag,
        ) = NoisyPrefixOptimizer._insert_adversarial_prefix(
            batch_attention_mask, batch_answer_flags, batch_reward_flags, batch_embeddings, adversarial_prefix
        )
        tokenized_batch.data["inputs_embeds"] = prefixed_embeds
        tokenized_batch.data["attention_mask"] = prefixed_mask
        tokenized_batch.data["answer_flag"] = prefixed_ans_flag
        tokenized_batch.data["reward_flags"] = prefixed_reward_flag
        tokenized_batch = tokenized_batch.to(self.gpu_id)

        # FORWARD PASS
        model_output = self.reward_model(
            **tokenized_batch,
            return_probs=True,
        )

        # LOSS CALCULATION
        reward_values = model_output[2][tokenized_batch.data["reward_flags"].bool()]
        mean_reward = reward_values.mean()
        nlr_loss = -torch.log(reward_values).mean()

        probs = torch.softmax(self.adversarial_logits, dim=-1)
        log_probs = (probs + 1e-12).log()
        entropy_total = -(probs * log_probs).sum()

        H_mean = entropy_total / self.hparams.NUM_PREFIXES
        H_norm = H_mean / math.log(self.vocabulary_size)
        
        lambda_t = self._get_lambda()
        H_penalty = lambda_t * H_mean

        attack_loss = nlr_loss + H_penalty

        with torch.no_grad():
            # MAKE MAGIC TOKENS
            hard_prefix = self._make_adversarial_prefix(hard=True)
            
            # INSERT MAGIC TOKENS (hard)
            (
                hard_embeds,
                hard_mask,
                hard_ans_flag,
                hard_reward_flag,
            ) = NoisyPrefixOptimizer._insert_adversarial_prefix(
                batch_attention_mask, batch_answer_flags, batch_reward_flags, batch_embeddings, hard_prefix
            )
            tokenized_batch.data["inputs_embeds"] = hard_embeds
            tokenized_batch.data["attention_mask"] = hard_mask
            tokenized_batch.data["answer_flag"] = hard_ans_flag # not necessary
            tokenized_batch.data["reward_flags"] = hard_reward_flag # not necessary
            tokenized_batch = tokenized_batch.to(self.gpu_id)

            # FORWARD PASS
            hard_output = self.reward_model(
                **tokenized_batch,
                return_probs=True,
            )
            
            # REWARD CALCULATION
            discrete_reward_values = hard_output[2][tokenized_batch.data["reward_flags"].bool()]
            discrete_reward = discrete_reward_values.mean()

            # OTHER METRICS
            avg_max_p = probs.max(dim=-1).values.mean()
            lambda_t_det = torch.tensor(lambda_t, device=self.gpu_id)

        # BACKPROPAGATION
        attack_loss.backward()

        # OPTIMIZER STEP
        grad_norm = self._optimizer_step()

        # LOGGING
        self._log_metrics(
            # Loss components (for plotting)
            attack_loss=attack_loss.detach(),
            nlr_loss=nlr_loss.detach(),
            H_penalty=H_penalty.detach(),
            
            # Reward metrics (for plotting)
            soft_reward=mean_reward.detach(),
            discrete_reward=discrete_reward.detach(),
            
            # Discreteness metrics (for plotting)
            H_norm=H_norm.detach(),
            avg_max_p=avg_max_p.detach(),
            
            # Optimization metrics (for plotting)
            grad_norm=grad_norm,
            lambda_t=lambda_t_det,
        )            
        
        self.global_step += 1
    
    def _log_metrics(self, **kwargs):
        """Aggregates metrics across GPUs and logs to console/history."""
        with torch.no_grad():
            # Collect tensors
            tensors = [t for t in kwargs.values() if isinstance(t, torch.Tensor)]
            
            # Average across GPUs
            for t in tensors:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t /= self.num_gpus

            if self.noisy:
                # 1. Add all metrics to history as floats
                metrics_for_history = {k: v.item() for k, v in kwargs.items()}
                self.metrics_series.append(metrics_for_history)

                # 2. Create a cleaner postfix for tqdm
                postfix_dict = {
                    "L_total": f"{kwargs['attack_loss'].item():.3f}",
                    "R_soft": f"{kwargs['soft_reward'].item():.3f}",
                    "R_hard": f"{kwargs['discrete_reward'].item():.3f}",
                    "L_nlr": f"{kwargs['nlr_loss'].item():.3f}",
                    "H_norm": f"{kwargs['H_norm'].item():.2f}",
                    "p_max": f"{kwargs['avg_max_p'].item():.2f}",
                    "grad": f"{kwargs['grad_norm'].item():.2e}",
                }
                
                if self.hparams.FULL_BATCH:
                    self.progress_bar.set_postfix(postfix_dict)
                else:
                    self.progress_bar.update(self.hparams.BATCH_SIZE * self.num_gpus)
                    self.progress_bar.set_postfix(postfix_dict)

    def _save_artifacts(self, final_logits_cpu: torch.Tensor):
        """Saves all run artifacts to the run directory."""
        print("\n--- Saving artifacts ---")
        
        # Save optimized logits
        opt_path = os.path.join(self.run_dir, "optimized_logits.pt")
        artifacts.save_logits(final_logits_cpu, opt_path)

        # Save token visualizations
        probs = torch.softmax(final_logits_cpu, dim=-1)
        artifacts.save_token_visualizations(probs, self.run_dir)

        # Save config
        artifacts.save_hyperparams(self.run_dir)

        # Save initial logits for comparison
        init_path = os.path.join(self.run_dir, "initial_logits.pt")
        artifacts.save_logits(self.initial_logits_cpu, init_path)

        # Save various training metrics
        metrics_csv = os.path.join(self.run_dir, "metrics.csv")
        metrics_png = os.path.join(self.run_dir, "metrics.png")
        artifacts.save_metrics(self.metrics_series, metrics_png, metrics_csv)
        print("--- Artifact saving complete ---")


# ===============================
# main entry point
# ===============================

def train(gpu_id, num_gpus):
    """Initializes the dataset and runs the trainer."""
    
    # 1. Initialize hyperparameters from config defaults
    hparams = cfg.Hyperparameters()
    
    # 2. Initialize the dataset
    # data = PRM800k("phase2_train.jsonl", hparams.DATA_SUBSET_SIZE)
    data = SingleQADataset(
        question="Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.",
        answer_steps=[
            "To determine the total weight of all Cindy's books, we need to calculate the weight of each book individually and then sum these weights.",
            "First, for the math and science books:\n- Each math book weighs 2 pounds.\n- Each science book weighs 2 pounds.\n- Cindy has 2 math books and 2 science books.\n- Total weight of math books: \\(2 \\text{ books} \\times 2 \\text{ pounds/book} = 4 \\text{ pounds}\\).\n- Total weight of science books: \\(2 \\text{ books} \\times 2 \\text{ pounds/book} = 4 \\text{ pounds}\\).\n- Combined weight of math and science books: \\(4 \\text{ pounds} + 4 \\text{ pounds} = 8 \\text{ pounds}\\).",
            "Second, for the French book:\n- The French book weighs 4 pounds.",
            "Third, for the English book:\n- The English book weighs 3 pounds.",
            "Fourth, for the history book:\n- The history book weighs twice as much as the English book.\n- Weight of the history book: \\(2 \\times 3 \\text{ pounds} = 6 \\text{ pounds}\\).",
            "Finally, for the total weight:\n- Sum of the weights of all the books: \\[ 8 \\text{ pounds} \\text{ (math and science)} + 4 \\text{ pounds} \\text{ (French)} + 3 \\text{ pounds} \\text{ (English)} + 6 \\text{ pounds} \\text{ (history)} = 21 \\text{ pounds} \\]",
            "Therefore, the total weight of the books Cindy is carrying is \\(\\boxed{21}\\) pounds.",
        ],
        size=hparams.DATA_SUBSET_SIZE,
    )
    
    # 3. Pass all dependencies to the trainer
    trainer = NoisyPrefixOptimizer(data, gpu_id, num_gpus, hparams)
    optimized_logits = trainer.run()


def main():
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = torch.cuda.current_device()
    torch.cuda.set_device(local_rank)

    train(rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
