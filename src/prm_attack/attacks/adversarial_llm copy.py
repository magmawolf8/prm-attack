import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork
from prm_attack.config import (
    SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
)


# Load language model
model_id = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model_device = torch.device("cuda:0")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(model_device).eval()

# Load PRM
prm_tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
prm_device = torch.device("cuda:1")
prm = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(prm_device).eval()

# check that they have the same encodings:
#print(tokenizer(".\n\n\n", return_tensors="pt"))
#print(prm_tokenizer._tokenizer(". ", return_tensors="pt"))
#exit()
question = "A class of 50 students has various hobbies. 10 like to bake, 5 like to play basketball, and the rest like to either play video games or play music. How many like to play video games if the number that like to play music is twice the number that prefer playing basketball?"
steps = [
  "To find out how many students like to play video games, let's start with the information given: There are 50 students in total. 10 students like to bake. 5 students like to play basketball. The number of students who like to play music is twice the number of students who like to play basketball.",
  "First, we find out how many students like to play music: \\[ \\text{Number of students who like to play music} = 2 \\times 5 = 10 \\]",
  "Now, we subtract the number of students who like to bake and play basketball from the total number of students to find out how many students like to play video games: \\[ \\text{Number of students who like to play video games} = \\text{Total students} - (\\text{Bake students} + \\text{Basketball students}) \\] \\[ \\text{Number of students who like to play video games} = 50 - (10 + 5) \\] \\[ \\text{Number of students who like to play video games} = 50 - 15 \\] \\[ \\text{Number of students who like to play video games} = 35 \\]",
  "So, there are 35 students who like to play video games."
]

inputs = tokenizer(question, return_tensors="pt")
inputs = {k: v.to(model_device) for k, v in inputs.items()}

with torch.no_grad(): # Generate pre-fill stuff
    out = model(**inputs, use_cache=True)
    logits = out.logits[:, -1, :]
    cache = out.past_key_values

q_inputs = prm_tokenizer.prepare_steps(question, steps)
q_inputs.to(prm_device)

with torch.no_grad(): # Generate default
    out = prm(**q_inputs, return_prob=True)
    default_reward = out.rewards[q_inputs.data["reward_flags"].bool()]

def pick_next_token(logits: torch.Tensor, generated_ids) -> int:
    # iterate through the logits from most to least probable.
    # take this id, and previous generated_ids, 
    # piece together the ids, including question and steps and give to PRM.
    index = torch.nonzero(q_inputs.data["answer_flag"][0])[0]
    mod_inputs = dict()
    mod_inputs["input_ids"] = q_inputs.data["input_ids"].tolist()
    mod_inputs["input_ids"] = [mod_inputs["input_ids"][0][:index] + generated_ids + [0] + prm_tokenizer.step_token_ids + mod_inputs["input_ids"][0][index:]]
    mod_inputs["input_ids"] = torch.tensor(mod_inputs["input_ids"]).to(prm_device)

    mod_inputs["attention_mask"] = torch.ones_like(mod_inputs["input_ids"]).to(prm_device)

    mod_inputs["reward_flags"] = torch.cat((torch.zeros(len(generated_ids) + 1 + len(prm_tokenizer.step_token_ids)).to(prm_device), q_inputs.data["reward_flags"][0]))

    mod_loc = index + len(generated_ids)

    _, indices = torch.sort(logits, dim=-1, descending=True)
    indices = indices[0]
    arg_max = indices[0]
    for i in indices:
        mod_inputs["input_ids"][0][mod_loc] = i
        out = prm(**mod_inputs, return_prob=True)
        modified_reward = out.rewards[0][mod_inputs["reward_flags"].bool()]
        if modified_reward[0] > default_reward[0]:
            return i.item(), modified_reward

    return arg_max, modified_reward

max_new_tokens = 200
generated_ids = []

with torch.no_grad():
    for _ in range(max_new_tokens):
        next_id, modified_reward = pick_next_token(logits, generated_ids)

        generated_ids.append(next_id)
        continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(continuation + "\n----------------------------")

        next_token = torch.tensor([[next_id]], device=model_device)

        out = model(
            input_ids=next_token,
            use_cache=True,
            past_key_values=cache,
        )
        logits = out.logits[:, -1, :]
        cache = out.past_key_values

        if next_id == tokenizer.eos_token_id or next_id == 13 or next_id == 624 or next_id == 382: # end of sentence
            break

continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(continuation)

print(default_reward)
print(modified_reward)