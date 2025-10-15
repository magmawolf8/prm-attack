# load language model
# load PRM
# load tokenizer
# use a single question + answer

# do one forward pass on the trajectory
# get the logits
# make the vector from the logits (or just extract from the end of the decoder stack)
# get mean of logits. Use this to do forward pass then backprop of PRM.
# append to the end of the sequence.

# damn I suck at ML research, couldn't even just put the tokens at the end

# anyways you do a forward pass, then backpropagation, to get derivative of end of step
# reward w.r.t. this amalgamation.
# the gradient

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork
from prm_attack.config import (
    SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
)

# load dataset
gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")

# load language model
model_id = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model_device = torch.device("cuda:0")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(model_device).eval()

# load PRM
prm_tokenizer = SkyworkTokenizerAPI(SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN)
prm_device = torch.device("cuda:1")
prm = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME).to(prm_device).eval()

ind = 0
print(gsm8k)