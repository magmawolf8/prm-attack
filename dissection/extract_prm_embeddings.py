# Is it feasible to extract all the text embeddings for every token?
# Perhaps I should only extract the text embeddings for the tokens in
# the dataset. However that limits the replacements I can do. If BERT
# for example, suggests a replacement that isn't in the domain, I would
# not be able to complete the replacement.

import access_prm
from transformers import AutoTokenizer
import torch
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

import string




NAME = "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
MODEL = "Skywork/" + NAME

ALLOWED_CHARS = set(
    string.ascii_letters +
    string.digits +
    string.punctuation
)
K = 1000




def contains_invalid_characters(s):
    return any(char not in ALLOWED_CHARS for char in s)

def plot_3d_scatter(x, y, z, labels):
    fig = go.Figure(
        data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3),
            text=labels,
            hoverinfo='text'
        ),
        layout=go.Layout(
            scene=dict(
                xaxis_title='Primary',
                yaxis_title='Secondary',
                zaxis_title='Tertiary'
            )
        )
    )
    pyo.plot(fig, filename='embedding_3d_plot.html', auto_open=False)


embed_hVec_vVec = access_prm.get_input_embeddings()

# filter embeddings whose corresponding tokens are not english characters or punctuation
prm_tokenizer_obj = AutoTokenizer.from_pretrained(
    MODEL, 
    trust_remote_code=True
)

valid_english_ind = []
valid_english_labels = []
for i in range(embed_hVec_vVec.shape[0]):
    text = prm_tokenizer_obj.decode(i)
    if not contains_invalid_characters(text):
        valid_english_ind.append(i)
        valid_english_labels.append(text)

embed_hVec_vVec = embed_hVec_vVec[valid_english_ind]

u, s, v = torch.pca_lowrank(embed_hVec_vVec, center=True)
embed_3Vec_vVec = torch.matmul(embed_hVec_vVec, v[:, :3])

embed_3Vec_vVec = embed_3Vec_vVec.cpu().detach().numpy()
X = embed_3Vec_vVec[:, 0]
Y = embed_3Vec_vVec[:, 1]
Z = embed_3Vec_vVec[:, 2]

plot_3d_scatter(X, Y, Z, valid_english_labels)


# while True:
#     word = input("word: ")
#     token = 

#     chosen_embed = embed_hVec_vVec[prm_tokenizer_obj.encode(word)[0]].unsqueeze(0)

#     chosen_norm = torch.nn.functional.normalize(chosen_embed, p=2, dim=1) # (1, hVec)
#     embed_norm = torch.nn.functional.normalize(embed_hVec_vVec, p=2, dim=1) # (vVec, hVec)

#     cos_simil = torch.matmul(chosen_norm, embed_norm.T) # inner product (1, vVec)
#     topk_simil, topk_ind = torch.topk(cos_simil, k=K, dim=1)
#     topk_simil = topk_simil.squeeze(0)
#     topk_ind = topk_ind.squeeze(0)

#     topk_embed_hVec_kVec = embed_hVec_vVec[topk_ind]

#     u, s, v = torch.pca_lowrank(embed_hVec_vVec, center=True)
#     topk_embed_3Vec_vVec = torch.matmul(topk_embed_hVec_kVec, v[:, :3])

#     topk_ind = topk_ind.cpu().detach().tolist()
#     tokens = list(map(lambda s: prm_tokenizer_obj.decode(s), topk_ind))
#     topk_embed_3Vec_vVec = topk_embed_3Vec_vVec.cpu().detach().numpy()
#     embedx_vVec = topk_embed_3Vec_vVec[:, 0]
#     embedy_vVec = topk_embed_3Vec_vVec[:, 1]
#     embedz_vVec = topk_embed_3Vec_vVec[:, 2]

#     scatter = go.Scatter3d(
#         x=embedx_vVec,
#         y=embedy_vVec,
#         z=embedz_vVec,
#         mode='markers',
#         marker=dict(size=3),
#         text=tokens,
#         hoverinfo="text+x+y+z"
#     )

#     layout = go.Layout(
#         scene=dict(
#             xaxis_title='Primary',
#             yaxis_title='Secondary',
#             zaxis_title='Tertiary'
#         )
#     )

#     fig = go.Figure(data=[scatter], layout=layout)

#     pyo.plot(fig, filename='embedding_3d_' + word + '.html', auto_open=False)

