# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
# python modules
import random
import time
import math
# tensor modules
import torch
import torch.nn.functional as F
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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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




NUM_EPOCHS = 3
BATCH_SIZE = 2
NUM_VECS = 7
LEARNING_RATE = 1e-2
SEED = 420
DATASET_SIZE = 2000

torch.manual_seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


# TODO 



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
    prefixed_attention_mask = torch.nn.functional.pad(input=inputs.data["attention_mask"], pad=(prefix_len, 0), value=1)
    prefixed_answer_flag = torch.nn.functional.pad(input=inputs.data["answer_flag"], pad=(0, prefix_len))
    prefixed_reward_flags = torch.nn.functional.pad(input=inputs.data["reward_flags"], pad=(prefix_len, 0))

    return prefixed_inputs_embeds, prefixed_attention_mask, prefixed_answer_flag, prefixed_reward_flags




def gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # sample from Gumbel distribution
    #gumbel_noise = -torch.empty_like(logits).exponential_().log()
    # gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20))

    # add Gumbel noise to logits
    # gumbel_logits = (logits + gumbel_noise) / temperature

    # apply softmax
    #return F.softmax(gumbel_logits, dim=-1)
    return F.softmax(logits / temperature, dim=-1)

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
    loader = DataLoader(train_prm800k, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=4, persistent_workers=True, collate_fn=collate_fn)

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
    net = net.to(rank).eval()

    embedding_layer = net.pretrained_model.model.embed_tokens.weight
    vocab_size = embedding_layer.shape[0]
    #logits = torch.nn.Parameter(torch.normal(0, (2/vocab_size)**0.5, (NUM_VECS, vocab_size), requires_grad=True, device=rank))
    logits = torch.nn.Parameter(torch.empty(NUM_VECS, vocab_size, device=rank))
    torch.nn.init.xavier_uniform_(logits)
    #optimizer = torch.optim.SGD([logits], lr=LEARNING_RATE, maximize=False)
    optimizer = torch.optim.Adam([logits], lr=LEARNING_RATE)

    start_temp = 1
    end_temp = 0.01

    steps = NUM_EPOCHS * DATASET_SIZE / BATCH_SIZE / world_size

    alpha = math.exp((math.log(end_temp) - math.log(start_temp))/steps)

    ema_loss = None
    ema_beta = 0.98

    i = 0
    if rank == 0:
        print("Training start")

    for epoch in range(NUM_EPOCHS):
        if rank == 0:
            pbar = tqdm(total=len(train_prm800k), desc=f"Epoch {epoch}")
        sampler.set_epoch(epoch)
        for batch in loader:

            probs = gumbel_softmax(logits, start_temp * pow(alpha, i))
            prefix = probs @ embedding_layer
            
            # prepare steps of the batch; tokenize etc.
            questions, answers = batch
            inputs = skywork_tokenizer_api.prepare_steps(questions, answers)

            # insert the continuous prefix
            inputs_embeds = embedding_layer[inputs.data["input_ids"]]
            inputs_embeds, attn_mask, answer_flag, reward_flags = insertPrefix(inputs, inputs_embeds, prefix)
            inputs.data["attention_mask"] = attn_mask
            inputs.data["answer_flag"] = answer_flag
            inputs.data["reward_flags"] = reward_flags
            inputs = inputs.to(rank)

            # run the model
            forward_output = net(**inputs, inputs_embeds=inputs_embeds, return_prob=True)

            # calculate the cost (want to maximize reward adversarially)
            masked_cost_fn = -torch.log(forward_output.rewards[inputs.data["reward_flags"].bool()])
            masked_cost_fn = masked_cost_fn.mean()

            # calculate the parallelized gradient
            masked_cost_fn.backward()
            dist.all_reduce(logits.grad, op=dist.ReduceOp.SUM)
            logits.grad /= world_size

            # step the optimizer
            optimizer.step()
            optimizer.zero_grad()

            #calculate diagnostic stuff
            if ema_loss is None:
                ema_loss = masked_cost_fn.item()
            else:
                ema_loss = ema_beta * ema_loss + (1 - ema_beta) * masked_cost_fn.item()

            i += 1

            if rank == 0:
                tokens = torch.argmax(probs, dim=-1)
                if rank == 0 and i % 5 == 0:
                    print(f"[Step {i}] logits mean: {logits.data.mean():.4f}, std: {logits.data.std():.4f}, max: {logits.data.max():.4f}")
                pbar.update(BATCH_SIZE * world_size)
                #pbar.set_postfix(cost=f"{masked_cost_fn.item():.4f}", temp=f"{start_temp * pow(alpha, i):.4f}", prefix=repr(skywork_tokenizer_api._tokenizer.decode(tokens)), value=f"{torch.max(probs, dim=-1)}")
                pbar.set_postfix(EMA_loss=f"{ema_loss:.4f}", max_prob=f"{torch.max(probs, dim=-1).values.tolist()}")

    torch.save(prefix, f"prefix_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_nvecs{NUM_VECS}_lr{LEARNING_RATE}_size{DATASET_SIZE}.pt")

    cleanup()




def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
