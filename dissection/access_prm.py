"""The goal of this program is to serve forward and backward propaga-
tion of the Skywork-o1-Open-PRM process reward model."""




from external.skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL
import torch




NAME = "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
MODEL = "Skywork/" + NAME
_PRM_OBJ = None
DEVICE = "cuda"




def get_input_embeddings():
    """Return input embeddings of the process reward model"""
    global _PRM_OBJ
    if _PRM_OBJ is None:
        _PRM_OBJ = PRM_MODEL.from_pretrained(MODEL).to(DEVICE)
        _PRM_OBJ.eval()
    return _PRM_OBJ.pretrained_model.model.embed_tokens.weight

def forward_reward_of(t_p_tVec_bVec: torch.LongTensor, 
                        attn_tVec_bVec: torch.BoolTensor,
                        **kwargs):
    """Calculate reward of the process reward model, given t_p_tVec_bVec, which is a matrix representing the tokens of multiple question-answer sequences."""
    global _PRM_OBJ
    if _PRM_OBJ is None:
        _PRM_OBJ = PRM_MODEL.from_pretrained(MODEL).to(DEVICE)
        _PRM_OBJ.eval()
    t_p_tVec_bVec = t_p_tVec_bVec.to(DEVICE)
    attn_tVec_bVec = attn_tVec_bVec.to(DEVICE)
    return _PRM_OBJ(
        input_ids=t_p_tVec_bVec, 
        attention_mask=attn_tVec_bVec,
        **kwargs
    )

def backward_gradients_of(r_bVec: torch.Tensor, 
                          embed_p_tVec_bVec: torch.Tensor):
    """Take the backward gradient of the reward vector with respect to the input vector embeddings."""
    assert r_bVec.dim() == 1, "reward vector must be 1D"
    r_bVec.backward(
        gradient=torch.ones(r_bVec.shape).to(DEVICE), 
        inputs=embed_p_tVec_bVec
    )
    return embed_p_tVec_bVec.grad