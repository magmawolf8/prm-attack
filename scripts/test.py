# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, DEVICE
# tensor modules
import torch
# dataset modules
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

skywork_tokenizer_api = SkyworkTokenizerAPI(
    SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
)
net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
net = net.to(DEVICE).eval()
embedding_layer = net.pretrained_model.model.embed_tokens.weight

prefix = torch.load("prefix_epochs5_batch2_nvecs1_lr0.01_size2000.pt", weights_only=True).to(DEVICE)
# embeds_len = embedding_layer.shape[1]
# prefix = torch.normal(0, (2/embeds_len)**0.5, (15, embeds_len), device=DEVICE)

logits = prefix @ embedding_layer.T
tokens = torch.argmax(logits, dim=1)
print(skywork_tokenizer_api._tokenizer.decode(tokens))


def collate_fn(batch):
    questions, answers = zip(*batch)
    return list(questions), list(answers)

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

def test():
    test_prm800k = PRM800k("phase2_train.jsonl", 500)
    loader = DataLoader(test_prm800k, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    sum_unedited = 0
    sum_modified = 0

    for i, batch in tqdm(enumerate(loader)):
        questions, answers = batch

        inputs = skywork_tokenizer_api.prepare_steps(questions, answers)

        inputs_embeds = embedding_layer[inputs.data["input_ids"]]
        inputs_embeds, attn_mask, answer_flag, reward_flags = insertPrefix(inputs, inputs_embeds, prefix)

        inputs = inputs.to(DEVICE)

        with torch.no_grad():
            forward_unedited = net(**inputs, return_prob=True)
            forward_modified = net(input_ids=inputs.data["input_ids"], attention_mask=attn_mask, inputs_embeds=inputs_embeds, return_prob=True)
        masked_unedited = forward_unedited.rewards[inputs.data["reward_flags"].bool()]
        masked_modified = forward_modified.rewards[reward_flags.bool()]
        sum_unedited += masked_unedited.mean()
        sum_modified += masked_modified.mean()

    print(f"Reward without prefix: {sum_unedited/(i+1):.6f} Reward with prefix: {sum_modified/(i+1):.6f}")

if __name__ == "__main__":
    test()
# calculate average reward from dataset without prefix

# calculate average reward with prefix