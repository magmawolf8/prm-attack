# testing modules
import pytest
# dataset modules
from datasets import load_dataset




@pytest.fixture(scope="session")
def dataset(): # should load other data too
    data = load_dataset("Qwen/ProcessBench", split="gsm8k")
    return data


def test_feature_one(dataset):
    assert not dataset.empty

def test_feature_two(dataset):
    assert "important_column" in dataset.columns
