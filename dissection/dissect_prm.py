'''The goal of this program is to produce a function which can intake
the prm, the tokenizer, and a string of text representing a q and a,
and calculate the embeddings corresponding to the tokens in the str-
ing. A second function will clamp those values between 0 and 1.
Basically I want to investigate how the reward is extracted so I know
how to backpropagate the PRM later.'''




from datasets import load_dataset
import torch
import access_prm
from transformers import AutoTokenizer




NAME = "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
MODEL = "Skywork/" + NAME

STEP = "\n\n"
# it would be interesting to see how different step tokens affect the
# PRM's grade.




def p_token_of(prm_tokenizer_obj: AutoTokenizer,
                    q_bSeq: list[str], s_sSeq_bSeq: list[list[str]], ret_question_end=False):
    """Tokenize a batch of questions and respective answers"""
    # sSeq: sequence of steps
    # tSeq: token sequence <-> string
    # bSeq: batch sequence
    t_p_tSeq_bSeq = list(map(
        lambda q: prm_tokenizer_obj.encode(
            prm_tokenizer_obj.bos_token + q + '\n'
        ), 
        q_bSeq
    ))
    if ret_question_end == True:
        t_len_bSeq = list(map(lambda t_p_tSeq: len(t_p_tSeq), t_p_tSeq_bSeq))
    # English is combined together, then tokenized
    # Makes for more in-context tokenization
    steploc_sSeq_bSeq = []
    for s_sSeq, t_p_tSeq in zip(s_sSeq_bSeq, t_p_tSeq_bSeq):
        steploc_sSeq = []
        steploc_sSeq_bSeq.append(steploc_sSeq)
        for s in s_sSeq:
            t_p_tSeq.extend(prm_tokenizer_obj.encode(s + STEP))
            steploc_sSeq.append(len(t_p_tSeq) - 1)
    if ret_question_end == True:
        return t_p_tSeq_bSeq, steploc_sSeq_bSeq, t_len_bSeq
    else:
        return t_p_tSeq_bSeq, steploc_sSeq_bSeq

def sigmoid(x):
    return 1/(torch.exp(-x) + 1)

def p_reward_of(pad_token_id: int, t_p_tSeq_bSeq: list[list[int]], **kwargs):
    t_p_tVec_bVec = torch.nn.utils.rnn.pad_sequence(
        list(map(lambda t_p_tSeq: torch.LongTensor(t_p_tSeq), t_p_tSeq_bSeq)), 
        batch_first=True, 
        padding_value=pad_token_id
    )
    attn_tVec_bVec = (t_p_tVec_bVec != pad_token_id)
    return access_prm.forward_reward_of(
        t_p_tVec_bVec=t_p_tVec_bVec,
        attn_tVec_bVec=attn_tVec_bVec,
        **kwargs
    )

def t_step_r_of(prm_tokenizer_obj: AutoTokenizer, 
                q_bSeq: list[str], s_sSeq_bSeq: list[list[str]]):
    """Find the reward with respect to each step in a batch of question and respective answer."""
    t_p_tSeq_bSeq, steploc_sSeq_bSeq = p_token_of(prm_tokenizer_obj, 
                                                  q_bSeq, s_sSeq_bSeq)
    _, _, r_tVec_bVec = p_reward_of(
        prm_tokenizer_obj.pad_token_id,
        t_p_tSeq_bSeq,
        return_input_embeddings=False
    )
    step_r_sSeq_bSeq = []
    for b_i in range(len(steploc_sSeq_bSeq)):
        step_r_sSeq_bSeq.append(list(map(
            lambda steploc: sigmoid(r_tVec_bVec[b_i, steploc]).item(), 
            steploc_sSeq_bSeq[b_i]
        )))
    return step_r_sSeq_bSeq




def main() -> None:
    data_map_eSeq = load_dataset("Qwen/ProcessBench", split="gsm8k")
    prm_tokenizer_obj = AutoTokenizer.from_pretrained(
        MODEL, 
        trust_remote_code=True
    )

    question = list()
    answer = list()

    for data_map in data_map_eSeq:
        if "Steve" in data_map["problem"]:
            question.append(data_map["problem"])
            answer.append(data_map["steps"])

    questions = ["Steve is 60 years old. His wife is 4 years older than him. Their son is currently half as old as his mom and their son's wife is 3 years younger than her husband. How old is Steve's son's wife?",
                 "Steve is 60 years old. His wife is 4 years older than him. Their son is currently half as old as his mom and their son's wife is 3 years younger than her husband. How old is Steve's son's wife?",
                 "Steve is 60 years old. His wife is 4 years older than him. Their son is currently half as old as his mom and their son's wife is 3 years younger than her husband. How old is Steve's son's wife?"]
    answers = [["To find the age of Steve's son's wife, we need to follow these steps: First, find the age of Steve's wife: Since Steve is 60 years old, and his wife is 4 years older than him, Steve's wife's age = 60 + 4 = 64 years.", "Second, find the age of Steve's son: Since his son is half as old as his mom (who is 64 years old), Steve's son's age = 64 / 2 = 32 years.", "Third, find the age of Steve's son's wife: Since she is 3 years younger than her husband, Steve's son's wife's age = 32 - 3 = 29 years.", "Therefore, the age of Steve's son's wife is \\boxed{29}."],
               ["To find the age of Steve's son's wife, we need to follow these steps: First, find the age of Steve's wife: Since Steve is 60 years old, and Steve's wife is 4 years older than Steve, Steve's wife's age = 60 + 4 = 64 years.", "Second, find the age of Steve's son: Since Steve's son is half as old as Steve's son's mom (who is 64 years old), Steve's son's age = 64 / 2 = 32 years.", "Third, find the age of Steve's son's wife: Since Steve's son's wife is 3 years younger than Steve's son, Steve's son's wife's age = 32 - 3 = 29 years.", "Therefore, the age of Steve's son's wife is \\boxed{29}."],
               ["To find the age of his son's wife, we need to follow these steps: First, find the age of his wife: Since he is 60 years old, and his wife is 4 years older than him, his wife's age = 60 + 4 = 64 years.", "Second, find the age of his son: Since his son is half as old as his son's mom (who is 64 years old), his son's age = 64 / 2 = 32 years.", "Third, find the age of his son's wife: Since she is 3 years younger than her husband, his son's wife's age = 32 - 3 = 29 years.", "Therefore, the age of his son's wife is \\boxed{29}."]]

    step_rewards = t_step_r_of(prm_tokenizer_obj, questions, answers)
    print(step_rewards)

    # correct = []
    # incorrect = []

    # for data_map in data_map_eSeq:
    #     step_r_sSeq_bSeq = t_step_r_of(prm_tokenizer_obj, [data_map["problem"]], [data_map["steps"]])
    #     if data_map["final_answer_correct"]:
    #         correct.append(step_r_sSeq_bSeq[0][-1])
    #     else:
    #         incorrect.append(step_r_sSeq_bSeq[0][-1])

if __name__ == "__main__":
    main()
