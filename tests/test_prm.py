# configuration
from prm_attack.config import SKYWORK_MODEL_NAME, DEVICE
# skywork modules
from prm_attack.models.clearskywork import ClearSkywork
# testing modules
import pytest
# tensor modules
import torch
import torch.nn as nn




@pytest.fixture(scope="session")
def prm():
    net = ClearSkywork.from_pretrained(SKYWORK_MODEL_NAME)
    net = net.to(DEVICE).eval()
    return net

def test_prm_initialization(prm):
    assert isinstance(prm, nn.Module)

def test_prm_input_ids(gsm8k, tokenizer_api, prm):
    question = gsm8k[0]["problem"]
    answer_steps = gsm8k[0]["steps"]
    inputs = tokenizer_api.prepare_steps(question, answer_steps)
    inputs = inputs.to(DEVICE)
    forward_output = prm(**inputs, return_prob=True)
    assert all(r >= 0 and r <= 1 for r in forward_output.rewards[0])

def test_prm_batch_input_ids(gsm8k, tokenizer_api, prm):
    questions = list()
    batch_answers_steps = list()
    for i in range(5):
        questions.append(gsm8k[i]["problem"])
        batch_answers_steps.append(gsm8k[i]["steps"])
    inputs = tokenizer_api.prepare_steps(questions, batch_answers_steps)
    inputs = inputs.to(DEVICE)
    forward_output = prm(**inputs, return_prob=True)
    assert all(
        all(r >= 0 and r <= 1 for r in entry_rewards)
        for entry_rewards in forward_output.rewards
    )

def test_tokenizer_api_invalid_batch(gsm8k, tokenizer_api):
    with pytest.raises(TypeError):
        question = gsm8k[0]["problem"]
        answer_steps = gsm8k[0]["steps"]
        answer_steps = [answer_steps]
        tokenizer_api.prepare_steps(question, answer_steps)
    with pytest.raises(ValueError):
        questions = list()
        batch_answers_steps = list()
        for i in range(10):
            questions.append(gsm8k[i]["problem"])
            batch_answers_steps.append(gsm8k[i]["steps"])
        batch_answers_steps = batch_answers_steps[:9]
        tokenizer_api.prepare_steps(questions, batch_answers_steps)

def test_prm_inputs_embeds(gsm8k, tokenizer_api, prm):
    question = gsm8k[0]["problem"]
    answer_steps = gsm8k[0]["steps"]
    inputs = tokenizer_api.prepare_steps(question, answer_steps)
    inputs = inputs.to(DEVICE)
    forward_output = prm(**inputs, return_prob=False)

    inputs_embeds = forward_output.inputs_embeds[0]
    inputs_embeds[inputs.data["answer_flag"][0].bool()] = 0

    forward_o2 = prm(**inputs, inputs_embeds=forward_output.inputs_embeds)
    assert not torch.allclose(forward_output.rewards, forward_o2.rewards)

def test_prm_reward_sanity(gsm8k, tokenizer_api, prm):
    batch_size = 3
    incorrect_ans_end_rewards = list()
    correct_ans_end_rewards = list()
    for ind in range(0, len(gsm8k), batch_size):
        questions = list()
        batch_steps = list()
        batch_correct = list()
        for i in range(ind, min(ind + batch_size, len(gsm8k))):
            questions.append(gsm8k[i]["problem"])
            batch_steps.append(gsm8k[i]["steps"])
            batch_correct.append(gsm8k[i]["final_answer_correct"])

        raw = tokenizer_api.prepare_steps(questions, batch_steps)
        inputs = raw.to(DEVICE)

        with torch.no_grad():
            out = prm(**inputs, return_prob=True)

        for correct, reward, flags in zip(
            batch_correct, 
            out.rewards, 
            inputs["reward_flags"]
        ):
            end = reward[flags.bool()][-1].item()
            if correct:
                correct_ans_end_rewards.append(end)
            else:
                incorrect_ans_end_rewards.append(end)
    
    correct_avg_reward = sum(correct_ans_end_rewards) / len(correct_ans_end_rewards)
    incorrect_avg_reward = sum(incorrect_ans_end_rewards) / len(incorrect_ans_end_rewards)
    print(f"Correct average: {correct_avg_reward}    Incorrect average: {incorrect_avg_reward}")
    assert correct_avg_reward > incorrect_avg_reward

def test_cuda_oom(gsm8k, tokenizer_api, prm):
    question = gsm8k[0]["problem"]
    answer_steps = gsm8k[0]["steps"]
    inputs = tokenizer_api.prepare_steps(question, answer_steps)
    inputs = inputs.to(DEVICE)
    for _ in range(100):
        forward_output = prm(**inputs, return_prob=True)
    assert all(r >= 0 and r <= 1 for r in forward_output.rewards[0])