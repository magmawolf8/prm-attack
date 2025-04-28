'''The goal of this program is to produce a function which can intake
the prm, the tokenizer, and a string of text representing a q and a,
and calculate the embeddings corresponding to the tokens in the str-
ing. A second function will clamp those values between 0 and 1.
Basically I want to investigate how the reward is extracted so I know
how to backpropagate the PRM later.
The goal of this program is to produce a function which can intake
the prm, the tokenizer, and a string of text representing a q and a,
and calculate both the reward value for that string of text, and the
gradient of the reward value w.r.t. the input embeddings.'''




from datasets import load_dataset
import torch
import dissect_prm
import access_prm
from transformers import AutoTokenizer

from tqdm import tqdm
import pickle




NAME = "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
MODEL = "Skywork/" + NAME

BATCH_SIZE = 5
GRADIENT_NORM_THRESHOLD = 0




def sigmoid(x):
    return 1/(torch.exp(-x) + 1)

def p_grad_of(prm_tokenizer_obj: AutoTokenizer,
              t_p_tSeq_bSeq: list[list[int]]):
    _, _, r_tVec_bVec, embed_p_tVec_bVec = dissect_prm.p_reward_of(
        prm_tokenizer_obj.pad_token_id,
        t_p_tSeq_bSeq,
        return_input_embeddings=True
    )
    r_bVec = r_tVec_bVec[:, -1]
    return access_prm.backward_gradients_of(r_bVec, embed_p_tVec_bVec)

def replace_of(prm_tokenizer_obj: AutoTokenizer,
               t_p_tSeq_bSeq: list[list[int]], t_len_bSeq: list[int]):
    """Find replacements in the answer sequence for a question-answer batch, by calculating the gradient of the PRM reward with respect to each token embedding and following that gradient's direction"""
    logits_b, _, r_tVec_bVec, embed_p_tVec_bVec = dissect_prm.p_reward_of(
        prm_tokenizer_obj.pad_token_id,
        t_p_tSeq_bSeq,
        return_input_embeddings=True
    )
    r_bVec = r_tVec_bVec[:, -1]
    g_b = access_prm.backward_gradients_of(r_bVec, embed_p_tVec_bVec) # (batch_size, seq_length, h_size)
    u_b = access_prm.get_input_embeddings() # (batch_size, vocab_size, h_size)
    print("u_b shape", u_b.shape)
    # need to make v, which has shape (batch_size, seq_length, h_size)
    v_b = embed_p_tVec_bVec
    # the best replacement for a token has high placement in the PRM's
    # backbone model, and has almost no difference between it, and the
    # sum of the original token's embedding and the projection of the 
    # new token onto the gradient. if u is new token vector, v is orig-
    # inal token vector, and g is gradient of reward wrt v,
    # the function to apply is |v + <u,g>/<g,g> * g - u|
    # we can then multiply this with the actual logits to find our new
    # logits, which take both into account.
    # Also we may have to look for tokens whose <g,g> is high.
    changed_t_p_b = list()
    #for g, u, v, t_p_tSeq, t_len, logits in zip(g_b, u_b, v_b, t_p_tSeq_bSeq, t_len_bSeq, logits_b):
    g, u, v, t_p_tSeq, t_len, logits = g_b, u_b, v_b, t_p_tSeq_bSeq, t_len_bSeq, logits_b
    changed_t_p = list()
    changed_t_p_b.append(changed_t_p)
    for seq_i in range(len(t_p_tSeq)): # no need to go through padding.
        grad_norm = torch.linalg.vector_norm(g[seq_i])
        if grad_norm > GRADIENT_NORM_THRESHOLD and seq_i > t_len:
            best_token = None
            best_token_score = -float('inf')
            for u_i in range(len(u)):
                out = torch.linalg.vector_norm(v[seq_i] + torch.inner(u[u_i], g[seq_i])/grad_norm**2 * g[seq_i] - u[u_i], dim=1)
                out = logits[seq_i, u_i] * out
                if out > best_token_score:
                    best_token = u_i
                    best_token_score = out
            # out has shape (,)
            # it should be possible to make it have shape (seq_length, vocab_size) like logits
            # but the filter makes things difficult. I think it's better to select a new token based on
            # the current system.
            changed_t_p.append(best_token)
        else:
            changed_t_p.append(t_p_tSeq[seq_i])
    return changed_t_p

def t_sensitivity_of(prm_tokenizer_obj: AutoTokenizer, 
                  q_bSeq: list[str], s_sSeq_bSeq: list[list[str]]):
    """Gets average sensitivity of tokens.
    
    Finds a map between token value and sensitivity of the PRM to that
    token, averaged across the occurrence of tokens in the batch. Thus,
    tokens which occur more with great sensitivity in the sequence will
    have greater sensitivity overall.
    
    Args:
        prm_tokenizer_obj: An instance of the PRM tokenizer.
        q_bSeq: a batch sequence of questions.
        s_sSeq_bSeq: a batch sequence of step sequences of solutions c-
            orresponding to the questions.
    
    Returns:
        A dict mapping token values to the corresponding average sensi-
        tivity of the PRM to that token, which is a float.
    """
    t_p_tSeq_bSeq, steploc_sSeq_bSeq = dissect_prm.p_token_of(
        prm_tokenizer_obj, 
        q_bSeq, s_sSeq_bSeq
    ) # sometimes you need an extremely elegant solution.
    # Sometimes you need to glue two things together for
    # a functional hybrid...
    grad_embed_p_hVec_tVec_bVec = p_grad_of(prm_tokenizer_obj, t_p_tSeq_bSeq)
    nge_p_tVec_bVec = (grad_embed_p_hVec_tVec_bVec ** 2).sum(dim=-1)
    # norm squared along last dimension, for a tensor of shape (seq, batch)
    nge_p_tSeq_bSeq = nge_p_tVec_bVec.tolist()
    tokenMapSense = dict()
    tokenMapFreq = dict()
    for t_p_tSeq, nge_p_tSeq in zip(t_p_tSeq_bSeq, nge_p_tSeq_bSeq):
        for t_p, nge_p in zip(t_p_tSeq, nge_p_tSeq):
            if t_p in tokenMapSense:
                tokenMapSense[t_p] = tokenMapSense[t_p] + nge_p
                tokenMapFreq[t_p] = tokenMapFreq[t_p] + 1
            else:
                tokenMapSense[t_p] = nge_p
                tokenMapFreq[t_p] = 1
    for token in tokenMapSense:
        tokenMapSense[token] = tokenMapSense[token] / tokenMapFreq[token]
    return tokenMapSense




prm_tokenizer_obj = AutoTokenizer.from_pretrained(
    MODEL, 
    trust_remote_code=True
)

q_bSeq = []
s_sSeq_bSeq = []

for split in ["gsm8k", "math", "olympiadbench", "omnimath"]:
    data_map_eSeq = load_dataset("Qwen/ProcessBench", split="gsm8k")
    for data_map in tqdm(data_map_eSeq):
        q_bSeq.append(data_map["problem"])
        s_sSeq_bSeq.append(data_map["steps"])

b_len = len(q_bSeq)
tokenMapSense = dict()

for b_i in tqdm(range(0, b_len, BATCH_SIZE)):
    b_end_i = b_i + BATCH_SIZE if b_i + BATCH_SIZE < b_len else b_len
    tokenMapSense.update(
        t_sensitivity_of(
            prm_tokenizer_obj, 
            q_bSeq[b_i:b_end_i], 
            s_sSeq_bSeq[b_i:b_end_i]
        )
    )
with open('tokenMapSense.pickle', "wb") as f:
    pickle.dump(tokenMapSense, f)
