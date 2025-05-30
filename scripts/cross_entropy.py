# statistics class
from prm_attack.analysis.statistics import Statistics

# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN, DEVICE
# dataset modules
from datasets import load_dataset
# models modules
from prm_attack.models.skywork_tokenizer import SkyworkTokenizerAPI
from prm_attack.models.clear_skywork import ClearSkywork
gsm8k = load_dataset("Qwen/ProcessBench", split="gsm8k")
skywork_tokenizer_api = SkyworkTokenizerAPI(
        SKYWORK_MODEL_NAME, DEFAULT_STEP_TOKEN
    )
net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
net = net.to(DEVICE).eval()
stats = Statistics(gsm8k, skywork_tokenizer_api, net, batch_size=5)
stats.cross_entropy()
# 0.41789949741518695 for gsm8k unaltered