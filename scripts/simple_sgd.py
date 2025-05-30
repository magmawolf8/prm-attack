# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, DEVICE
# tensor modules
import torch
# dataset modules
from datasets import load_dataset
# models modules
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork

gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
skywork_tokenizer_api = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
net = net.to(DEVICE).eval()

question = gsm8k[0]["problem"]
answer_steps = gsm8k[0]["steps"]
print(question)
"""
Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  
On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  
On Saturday morning, the neighbors took back one third of the flamingos, painted them white, 
and put these newly painted white flamingos back out on Sue's front yard.  
Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. 
At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?
"""
print(answer_steps)
# immediate thing to do for the paper right now is:
#   1. filter interesting examples out of all the examples we have, create a dataset. Go through parquet files of original trajectory, modified trajectory, and filter out extreme cases, see if extreme cases have examples where something is wrong (if bias is not working as intended, throw it out).
#       Requires checking (top 100 or 200 examples where delta reward is maximum) by hand. Maybe divide the parquet files and double-check to make sure the data is good.
#       have not manually checked examples. Check that somehow magically (for example in the question shuffling), the question is still relevant to the solution, filter out these erroneous modifications.
#       For question removal, worth checking out if PRM reward is erroneously high at the start since there is no question. Or if the question is reiterated in the answer trajectory, the reward remains high.
#       Need a spreadsheet for assigning which parquets to do
#       Do question shuffle, question removal, and numerical value modification
"""
['eoatudseoadculroae, aoecudasroceus, aoseucdsoacdesunao, asceoduadcsnc, aoesucdoasenuc, To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. 
First, on Friday, the neighbors start with 18 pink plastic flamingos.', 
"On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, \\(1/3 \\times 18 = 6\\) flamingos are taken back. 
So, they have \\(18 - 6 = 12\\) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. 
Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has \\(12 + 6 = 18\\) pink flamingos and 6 white flamingos.", 
"On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. 
By the end of Sunday morning, Sue has \\(18 + 18 = 36\\) pink flamingos and still 6 white flamingos.", 
'To find the difference, subtract the number of white flamingos from the number of pink flamingos: \\(36 - 6 = 30\\). 
Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is \\(\\boxed{30}\\).']
"""

inputs = skywork_tokenizer_api.prepare_steps(question, answer_steps)
inputs = inputs.to(DEVICE)
forward_output = net(**inputs, return_prob=False)

lr = 1e-2
def step(inputs_embeds):
    forward_output = net(**inputs, inputs_embeds=inputs_embeds)
    forward_output.rewards.backward(gradient=inputs.data["reward_flags"], inputs=inputs_embeds)
    # 0
    return inputs_embeds + inputs_embeds.grad * lr

inputs_embeds = forward_output.inputs_embeds

for _ in range(100):
    inputs_embeds = step(inputs_embeds)

f = net(**inputs, inputs_embeds=inputs_embeds)

print(f.rewards[-1][-1])
# original reward: 3.12 (0.95)
# tensor(53.2645, device='cuda:0', grad_fn=<SelectBackward0>)

embedding_layer = net.pretrained_model.model.embed_tokens.weight

logits = inputs_embeds[0] @ embedding_layer.T
tokens = torch.argmax(logits, dim=1)
print(skywork_tokenizer_api._tokenizer.decode(tokens))
"""
<|im_start|>Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?
To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. 
First, on Friday, the neighbors start with 18 pink plastic flamingos.

On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, \(1/3 \times 18 = 6\) flamingos are taken back. 
So, they have \(18 - 6 = 12\) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. 
Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has \(12 + 6 = 18\) pink flamingos and 6 white flamingos.

On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. 
By the end of Sunday morning, Sue has \(18 + 18 = 36\) pink flamingos and still 6 white flamingos.

To find the difference, subtract the number of white flamingos from the number of pink flamingos: \(36 - 6 = 30\). 
Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is \(\boxed{30}\).
"""