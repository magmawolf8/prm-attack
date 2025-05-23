"""Contains various shared dataclasses and configurations for the serv-
ices. Keeps communication between modules modular."""




# python modules
from dataclasses import dataclass
# tensor modules
import torch




@dataclass
class ForwardOutput:
    inputs_embeds: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor
    rewards: torch.Tensor


SKYWORK_MODEL_NAME = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_STEP_TOKEN = "\n\n"