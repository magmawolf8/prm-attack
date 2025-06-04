# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
# python modules
import random
import time
import pickle
# tensor modules
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
# dataset modules
import json
# models modules
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork
# util modules
from tqdm import tqdm




class PRM800k(Dataset):
    def __init__(self, jsonl_path, size):
        self.samples = []
        with open(jsonl_path, 'r') as f:
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




NUM_EPOCHS = 5
BATCH_SIZE = 2
NUM_VECS = 1
LEARNING_RATE = 1e-2
seed = 420
DATASET_SIZE = 2000
# TODO add validation "loss" (average reward on test dataset)
# TODO should we do the log of the success probability?
# TODO pickle the starting random vector
# TODO pickle is broken

torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)




skywork_tokenizer_api = SkyworkTokenizerAPI(
    SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
)

def insertPrefix(inputs, inputs_embeds, prefix):
    prefix_len = prefix.shape[0]

    batch_inputs_embeds = list()
    for embed, af in zip(inputs_embeds, inputs.data["answer_flag"]):
        index = torch.nonzero(af)[0]
        batch_inputs_embeds.append(torch.vstack((embed[:index], prefix, embed[index:])))

    prefixed_inputs_embeds = torch.stack(batch_inputs_embeds)
    prefixed_attention_mask = torch.nn.functional.pad(input=inputs.data["attention_mask"], pad=(prefix_len, 0))
    prefixed_answer_flag = torch.nn.functional.pad(input=inputs.data["answer_flag"], pad=(prefix_len, 0))
    prefixed_reward_flags = torch.nn.functional.pad(input=inputs.data["reward_flags"], pad=(prefix_len, 0))

    return prefixed_inputs_embeds, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags




def collate_fn(batch):
    questions, answers = zip(*batch)
    return list(questions), list(answers)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    train_prm800k = PRM800k("phase2_train.jsonl", DATASET_SIZE)
    sampler = DistributedSampler(train_prm800k, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_prm800k, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=4, collate_fn=collate_fn, persistent_workers=True)

    if rank == 0:
        print("Loading dataset workers...")
        start = time.perf_counter()
    _ = next(iter(loader))
    if rank == 0:
        end = time.perf_counter()
        print(f"Loading dataset workers took {(end - start):.1f} seconds")

    net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
    for param in net.parameters():
        param.requires_grad = False
    net = net.to(rank).train()

    embedding_layer = net.pretrained_model.model.embed_tokens.weight
    embeds_len = embedding_layer.shape[1]
    prefix = torch.nn.Parameter(torch.normal(0, (2/embeds_len)**0.5, (NUM_VECS, embeds_len), requires_grad=True, device=rank))
    optimizer = torch.optim.SGD([prefix], lr=LEARNING_RATE, maximize=True)

    for epoch in range(NUM_EPOCHS):
        if rank == 0:
            pbar = tqdm(total=len(train_prm800k), desc=f"Epoch {epoch}")
        sampler.set_epoch(epoch)
        for batch in loader:

            questions, answers = batch

            inputs = skywork_tokenizer_api.prepare_steps(questions, answers)

            inputs_embeds = embedding_layer[inputs.data["input_ids"]]
            inputs_embeds, attn_mask, answer_flag, reward_flags = insertPrefix(inputs, inputs_embeds, prefix)
            # decode to English again to observe vectors
            # logits = inputs_embeds[0] @ embedding_layer.T
            # tokens = torch.argmax(logits, dim=1)
            # skywork_tokenizer_api._tokenizer.decode(tokens))

            inputs.data["attention_mask"] = attn_mask
            inputs.data["answer_flag"] = answer_flag
            inputs.data["reward_flags"] = reward_flags
            inputs = inputs.to(rank)

            forward_output = net(**inputs, inputs_embeds=inputs_embeds, return_prob=True)

            masked_gain = torch.log(forward_output.rewards[inputs.data["reward_flags"].bool()])
            masked_gain = masked_gain.mean()

            masked_gain.backward()

            dist.all_reduce(prefix.grad, op=dist.ReduceOp.SUM)
            prefix.grad /= world_size

            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                pbar.update(BATCH_SIZE * world_size)
                pbar.set_postfix(gain=f"{masked_gain.item():.4f}")

    torch.save(prefix, f"prefix_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_nvecs{NUM_VECS}_lr{LEARNING_RATE}_size{DATASET_SIZE}.pt")

    cleanup()




def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    # use cross-entropy, but positive instead of negative
# need to make object which experiments with the output of the tokenizer
# maybe I subclass the skywork tokenizer API.
# then another object does backpropagation after getting ForwardOutput.
# actually we probably don't need a separate object, instead directly implement it in attack/

if __name__ == "__main__":
    main()