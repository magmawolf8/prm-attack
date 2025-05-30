# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, DEVICE
# testing modules
import pytest
# dataset modules
from datasets import load_dataset
# models modules
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork




@pytest.fixture(scope="session")
def gsm8k(): # should load other data too, not just gsm8k
    gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
    return gsm8k

@pytest.fixture(scope="session")
def tokenizer_api():
    skywork_tokenizer_api = SkyworkTokenizerAPI(
        SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
    )
    return skywork_tokenizer_api

@pytest.fixture(scope="session")
def prm():
    net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
    net = net.to(DEVICE).eval()
    return net