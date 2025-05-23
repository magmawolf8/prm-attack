# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
# testing modules
import pytest
# dataset modules
from datasets import load_dataset
# models modules
from prm_attack.models.skyworktokenizer import SkyworkTokenizerAPI




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