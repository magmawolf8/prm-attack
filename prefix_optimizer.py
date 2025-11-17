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

class PrefixOptimizer:
    """
    Manages the core computation for adversarial prefix optimization.
    This class contains NO logging, I/O, or other side effects.
    """

    def __init__(self,
                 dataset: Dataset,
                 gpu_id: int,
                 num_gpus: int,
                 hparams: cfg.Hyperparameters):
        
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.data = dataset
        self.hparams = hparams
        
        # Core state variables
        self.sampler = None
        self.data_loader = None
        self.skywork_tokenizer = None
        self.reward_model = None
        self.token_embedding_layer = None
        self.vocabulary_size = 0
        self.adversarial_logits = None
        self.optimizer = None
        self.steps_per_epoch = 0
        self.total_steps = 0
        self.global_step = 0

        # Setup
        self._setup_data()
        self._setup_model_and_opt()
        self._setup_schedule()

    def _setup_data(self):
        """Initializes the Sampler and DataLoader from the provided Dataset."""
        self.sampler = DistributedSampler(
            self.data,
            num_replicas=self.num_gpus,
            rank=self.gpu_id,
            shuffle=True,
        )

        if self.hparams.FULL_BATCH:
            local_batch_size = math.ceil(len(self.data) / self.num_gpus)
        else:
            local_batch_size = self.hparams.BATCH_SIZE

        self.data_loader = DataLoader(
            self.data,
            batch_size=local_batch_size,
            sampler=self.sampler,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            collate_fn=collate_into_batch,
        )

    def _setup_model_and_opt(self):
        """Loads the model, tokenizer, and creates optimizer and logits."""
        self.skywork_tokenizer = SkyworkTokenizer(
            self.hparams.SKYWORK_MODEL_NAME, self.hparams.STEP_TOKEN
        )
        self.reward_model = PRM_MODEL.from_pretrained(
            self.hparams.SKYWORK_MODEL_NAME,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float16
        ).to(self.gpu_id).eval()
        
        self.token_embedding_layer = self.reward_model.pretrained_model.model.embed_tokens.weight
        self.vocabulary_size = self.token_embedding_layer.shape[0]

        self.adversarial_logits = torch.nn.Parameter(
            torch.randn(self.hparams.NUM_PREFIXES, self.vocabulary_size, device=self.gpu_id)
        )

        self.optimizer = torch.optim.Adam(
            [self.adversarial_logits], lr=self.hparams.LEARNING_RATE, maximize=False
        )

    def _setup_schedule(self):
        """Sets up step counts."""
        if self.hparams.FULL_BATCH:
            self.steps_per_epoch = 1
        else:
            self.steps_per_epoch = math.ceil(
                len(self.data) / (self.hparams.BATCH_SIZE * self.num_gpus)
            )
        
        self.total_steps = max(1, self.steps_per_epoch * self.hparams.NUM_EPOCHS)
        self.global_step = 0

    def _get_lambda(self) -> float:
        """Calculates the entropy regularization weight for the current step."""
        t = min(1.0, self.global_step / self.total_steps)
        cos_t = 0.5 * (1 - math.cos(math.pi * t))
        return (1 - cos_t) * self.hparams.MIN_LAMBDA + cos_t * self.hparams.MAX_LAMBDA

    # hooks
    def on_train_start(self): pass
    def on_epoch_start(self, epoch: int): pass
    def on_epoch_end(self, epoch: int): pass
    def on_train_end(self, final_logits: torch.Tensor): pass
    
    def run(self) -> torch.Tensor:
        """
        Runs the main training loop and returns the optimized logits.
        """
        self.on_train_start()

        for epoch in range(self.hparams.NUM_EPOCHS):
            self.on_epoch_start(epoch)
            self.sampler.set_epoch(epoch) # DDP requirement
            
            for batch in self.data_loader:
                self.run_step(batch)
            
            self.on_epoch_end(epoch)

        dist.barrier()
        final_logits_cpu = self.adversarial_logits.detach().cpu()
        self.on_train_end(final_logits_cpu)
        
        return final_logits_cpu

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

        # MAKE MAGIC TOKENS (soft)
        adversarial_prefix = self._make_adversarial_prefix(hard=False)

        # INSERT MAGIC TOKENS
        (
            prefixed_embeds,
            prefixed_mask,
            prefixed_ans_flag,
            prefixed_reward_flag,
        ) = self._insert_adversarial_prefix(
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

        # BACKPROPAGATION
        attack_loss.backward()

        # OPTIMIZER STEP
        self._optimizer_step()
 
        self.global_step += 1
    
    def _make_adversarial_prefix(self, hard: bool) -> torch.Tensor:
        if hard:
            probs = F.one_hot(
                torch.argmax(self.adversarial_logits, dim=-1),
                num_classes=self.vocabulary_size
            ).type(torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float16)
        else:
            probs = F.gumbel_softmax(
                self.adversarial_logits, tau=self.hparams.TAU, hard=False, dim=-1
            ).type(torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float16)
        return probs @ self.token_embedding_layer

    @staticmethod
    def _insert_adversarial_prefix(batch_attention_mask, batch_answer_flags, batch_reward_flags, batch_embeddings, adversarial_prefix):
        prefix_length = adversarial_prefix.shape[0]
        batch_size = batch_embeddings.shape[0]
        device = batch_embeddings.device

        zeros_for_prefix = torch.zeros(prefix_length, dtype=torch.long, device=device)

        processed_embeddings_list = []
        processed_answer_flags_list = []
        processed_reward_flags_list = []

        for i in range(batch_size):
            sample_embedding = batch_embeddings[i]

            answer_flag_vector = batch_answer_flags[i].to(device)
            reward_flags_vector = batch_reward_flags[i].to(device)

            start_insertion_idx = torch.nonzero(answer_flag_vector, as_tuple=True)[0][0]
            end_insertion_idx = torch.nonzero(reward_flags_vector, as_tuple=True)[0][-1]

            new_embedding = torch.vstack(
                (
                    sample_embedding[:start_insertion_idx],
                    adversarial_prefix,
                    sample_embedding[start_insertion_idx:end_insertion_idx],
                    adversarial_prefix,
                    sample_embedding[end_insertion_idx:],
                )
            )
            processed_embeddings_list.append(new_embedding)

            new_answer_flag = torch.cat(
                (
                    answer_flag_vector[:start_insertion_idx],
                    zeros_for_prefix,
                    answer_flag_vector[start_insertion_idx:end_insertion_idx],
                    zeros_for_prefix,
                    answer_flag_vector[end_insertion_idx:],
                )
            )
            processed_answer_flags_list.append(new_answer_flag)

            new_reward_flag = torch.cat(
                (
                    reward_flags_vector[:start_insertion_idx],
                    zeros_for_prefix,
                    reward_flags_vector[start_insertion_idx:end_insertion_idx],
                    zeros_for_prefix,
                    reward_flags_vector[end_insertion_idx:],
                )
            )
            processed_reward_flags_list.append(new_reward_flag)

        prefixed_batch_embeddings = torch.stack(processed_embeddings_list)
        prefixed_answer_flag = torch.stack(processed_answer_flags_list)
        prefixed_reward_flags = torch.stack(processed_reward_flags_list)

        total_added_length = 2 * prefix_length
        prefixed_attention_mask = F.pad(
            input=batch_attention_mask,
            pad=(total_added_length, 0),
            value=1,
        )

        return (
            prefixed_batch_embeddings,
            prefixed_attention_mask,
            prefixed_answer_flag,
            prefixed_reward_flags,
        )

    def _optimizer_step(self) -> torch.Tensor:
        """Performs gradient reduction and optimizer step."""
        dist.all_reduce(self.adversarial_logits.grad, op=dist.ReduceOp.SUM)
        self.adversarial_logits.grad /= self.num_gpus
        
        grad_norm = torch.norm(self.adversarial_logits.grad, dim=-1).mean().detach()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return grad_norm



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
    trainer = PrefixOptimizer(data, gpu_id, num_gpus, hparams)
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
