import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from skywork_tokenizer import SkyworkTokenizerAPI
from skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL
from config import (
    SKYWORK_MODEL_NAME, STEP_TOKEN
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
prm_tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, STEP_TOKEN)
prm_device = torch.device("cuda:1")
prm = PRM_MODEL.from_pretrained(SKYWORK_MODEL_NAME).to(prm_device).eval()

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
#question = "Avery needs to buy a 3 piece place setting (dinner & salad plate and a bowl) for her holiday dinner.  She’s having 12 people over for dinner.  If the dinner plates cost $6.00 each and bowls each cost $5.00 and the salad plates cost $4.00, how much will she spend on place settings?"
#steps = [
#  "To determine how much Avery will spend on place settings for her holiday dinner, we need to calculate the total cost of each type of plate and then sum these costs.",
#  "First, let's break down the costs:\nFirst, dinner plates: There are 3 dinner plates. The cost per dinner plate is $6.00. Therefore, the total cost for dinner plates is 3 plates × $6.00/plate = $18.00.",
#  "Second, salad plates: There are 3 salad plates. The cost per salad plate is $4.00. Therefore, the total cost for salad plates is 3 plates × $4.00/plate = $12.00.",
#  "Third, bowls: There is 1 bowl. The cost per bowl is $5.00. Therefore, the total cost for bowls is 1 bowl × $5.00/bowl = $5.00.",
#  "Next, we add up all the individual costs:\nThe total cost for dinner plates is $18.00. The total cost for salad plates is $12.00. The total cost for bowls is $5.00.",
#  "Total cost for all place settings = $18.00 + $12.00 + $5.00 = $35.00.",
#  "Therefore, Avery will spend \\boxed{35} dollars on place settings for her holiday dinner."
#]
#question = "Pat has a flower bed that is 111 feet long.  Pat wants to fill her flower bed with plants.  Pat's flowers grow 12 inches wide so she needs to leave 1.5 feet between every plant.  Pat already owns 17 flowers.  Each flowering plant costs $6 at the store,  how much money will Pat spend at the store to fill up her flower bed?"
#steps = [
#  "To determine how much money Pat will spend on flowers to fill her flower bed, we need to follow these steps: First, calculate the total length needed for the plants. The flower bed is 111 feet long. There needs to be 1.5 feet of space between each plant.",
#  "Second, calculate the number of spaces required. If there are 1.5 feet of space between each plant, we divide the total length by this spacing: \\[ \\frac{111 \\text{ feet}}{1.5 \\text{ feet/pace}} = 74 \\text{ paces} \\] Since the first plant starts at the beginning of the bed, we add 1 to the number of paces to get the total number of plants: \\[ 74 + 1 = 75 \\text{ plants} \\]",
#  "Third, determine the cost of the flowers. Each flower costs $6. Pat already owns 17 flowers, so she needs to buy: \\[ 75 \\text{ plants} - 17 \\text{ owned plants} = 58 \\text{ additional plants} \\] The cost for 58 plants is: \\[ 58 \\times 6 = 348 \\text{ dollars} \\]",
#  "Therefore, the total amount of money Pat will spend at the store to fill up her flower bed is \\(\\boxed{348}\\)."
#]

m_inputs = prm_tokenizer.prepare_steps(question, steps)
m_inputs.to(model_device)

q_inputs = prm_tokenizer.prepare_steps(question, steps)
q_inputs.to(prm_device)

with torch.no_grad(): # Generate pre-fill stuff
    out = model(**m_inputs, use_cache=True)
    logits = out.logits[:, -1, :]
    cache = out.past_key_values

with torch.no_grad(): # Generate default
    out = prm(**q_inputs, return_prob=True)
    default_reward = out[2][q_inputs.data["reward_flags"].bool()]

def pick_next_token(logits: torch.Tensor, generated_ids) -> int:
    # iterate through the logits from most to least probable.
    # take this id, and previous generated_ids, 
    # piece together the ids, including question and steps and give to PRM.
    mod_inputs = dict()
    mod_inputs["input_ids"] = q_inputs.data["input_ids"].tolist()
    mod_inputs["input_ids"] = [mod_inputs["input_ids"][0] + generated_ids + [0] + prm_tokenizer.step_token_ids]
    mod_inputs["input_ids"] = torch.tensor(mod_inputs["input_ids"]).to(prm_device)

    mod_inputs["attention_mask"] = torch.ones_like(mod_inputs["input_ids"]).unsqueeze(0).to(prm_device)

    mod_inputs["reward_flags"] = torch.cat((q_inputs.data["reward_flags"][0], torch.zeros(len(generated_ids) + 1 + len(prm_tokenizer.step_token_ids)).to(prm_device)))
    mod_inputs["reward_flags"][-1] = 1 # flagging the last step, as another place to take reward.

    mod_loc = len(mod_inputs["input_ids"][0]) - len(prm_tokenizer.step_token_ids) - 1

    _, indices = torch.topk(logits, 80, dim=-1)
    indices = indices[0] # take a row view
    arg_max = None
    m = None
    for i in indices:
        mod_inputs["input_ids"][0, mod_loc] = i
        out = prm(**mod_inputs, return_probs=True)
        modified_reward = out[2][0][mod_inputs["reward_flags"].bool()]
        if m is None or modified_reward[-1] > m[-1]:
            m = modified_reward
            arg_max = i
    return arg_max, m

max_new_tokens = 25
generated_ids = []


print("default stepwise reward:", default_reward)

with torch.no_grad():
    for _ in range(max_new_tokens):
        next_id, modified_reward = pick_next_token(logits, generated_ids)
        print(str(modified_reward.tolist()) + "\n############################")

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

ids = q_inputs.data["input_ids"].tolist()
ids = ids[0] + generated_ids + prm_tokenizer.step_token_ids

mod_inputs = dict()
mod_inputs["input_ids"] = q_inputs.data["input_ids"].tolist()
mod_inputs["input_ids"] = [mod_inputs["input_ids"][0] + generated_ids + prm_tokenizer.step_token_ids]
mod_inputs["input_ids"] = torch.tensor(mod_inputs["input_ids"]).to(prm_device)

mod_inputs["attention_mask"] = torch.ones_like(mod_inputs["input_ids"]).to(prm_device)

mod_inputs["reward_flags"] = torch.cat((q_inputs.data["reward_flags"][0], torch.zeros(len(generated_ids) + len(prm_tokenizer.step_token_ids)).to(prm_device)))
mod_inputs["reward_flags"][-1] = 1

with torch.no_grad(): # Generate mod
    out = prm(**mod_inputs, return_prob=True)
    regen_modified_reward = out[2][0][mod_inputs["reward_flags"].bool()]

print("regenerated, modified stepwise reward", regen_modified_reward)

continuation = tokenizer.decode(ids, skip_special_tokens=True)
print(continuation)
print(repr(continuation))

print("modified stepwise reward", modified_reward)
print(generated_ids)