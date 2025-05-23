"""Model wrapper. Provides forward and backward pass on the PRM, and m-
anages to directly inject embeddings into the PRM's backbone LLM. Viol-
ates the abstraction barrier to set input embeddings and get the gradi-
ent with respect to the input embeddings. Also converts mixtures of di-
screte tokens and continuous vector embeddings into uniform vector emb-
eddings, and does the opposite too."""




# configuration
from prm_attack.config import ForwardOutput
# tensor modules
from .skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL
import torch




class ClearSkywork(PRM_MODEL):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)

    def forward(self, input_ids: torch.LongTensor, 
                      attention_mask: torch.BoolTensor,
                      inputs_embeds=None,
                      return_prob: bool = False,
                      **kwargs) -> ForwardOutput:
        lm_output = None
        if inputs_embeds is None:
            lm_output = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
        else:
            lm_output = self.pretrained_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )

        lm_logits = lm_output.logits
        loss = lm_output.loss
        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        last_hidden_state = (((
                    lm_output
                ).hidden_states[-1]
            ).to(self.v_head.summary.weight.device)
        )
        value = self.v_head(last_hidden_state).squeeze(-1)
        if return_prob:
            value = torch.nn.functional.sigmoid(value)

        return ForwardOutput(
            inputs_embeds=lm_output.hidden_states[0],
            logits=lm_logits, 
            loss=loss,
            rewards=value
        )


# class PRMAPI:
#     def __init__(self, model_name: str, device: str | torch.device = "cuda"):
#         self.model_name = model_name
#         self.device = torch.device(device)
#         self._net = _TransparentPRM.from_pretrained(model_name)
#         self._net = self._net.to(self.device).eval()

#     def forward_reward_cont(self, input_embeds: torch.Tensor,
#                                   attention_mask: torch.BoolTensor,
#                                   ) -> ForwardOutput:
#         input_embeds = input_embeds.to(self.device)
#         attention_mask = attention_mask.to(self.device)
#         return self._net(
#             input=input_embeds,
#             attention_mask=attention_mask,
#             skip_embedding_layer=True
#         )

#     def forward_reward(self, input_ids: torch.LongTensor,
#                              attention_mask: torch.BoolTensor,
#                              ) -> ForwardOutput:
#         input_ids = input_ids.to(self.device)
#         attention_mask = attention_mask.to(self.device)
#         return self._net(
#             input=input_ids,
#             attention_mask=attention_mask
#         )

#     def forward_reward_prob(self, input_ids: torch.LongTensor,
#                                   attention_mask: torch.BoolTensor,
#                                   ) -> ForwardOutput:
#         input_ids = input_ids.to(self.device)
#         attention_mask = attention_mask.to(self.device)
#         return self._net(
#             input=input_ids,
#             attention_mask=attention_mask,
#             return_prob=True
#         )

#     def backward_gradients(self, forward_output: ForwardOutput, 
#                                  select: Callable, 
#                                  grad_wrt_self: Callable) -> torch.Tensor:
#         """We could use the backward gradient function to do a little
#         More logic regarding forward and backward passes, and pass back
#         a BackwardOutput dataclass."""
#         # the gradient wrt self should be uniform across batches, but
#         # it's not necessary to be uniform within (across tokens).
#         # should use tokenizer to select reward etc.? Does that make sense?
#         # basically, the tokenizer is aware of steps and stuff. We shouldn't
#         # pass the tokenizer into the PRM API, but they should be linked in
#         # the main program in some way and the backward_gradients should get
#         # a forward_output with all of the items necessary to complete the
#         # gradient.
#         selected_rewards = select(forward_output)
#         uniform = torch.ones(selected_rewards.shape).to(self.device)
#         selected_rewards.backward(
#             gradient=uniform, 
#             inputs=forward_output.input_embeds
#         )
#         return BackwardOutput(
#             gradient=
#             input_embeds=input_embeds,
#             logits=None
#         )
    
#     def get_input_embeds(self, input_ids: torch.LongTensor) -> ForwardOutput:
#         embeds = self._net.pretrained_model.model.embed_tokens(input_ids)
#         return ForwardOutput(
#             input_embeds=embeds,
#             logits=None,
#             rewards=None
#         )

#     def get_logits(self, input_embeds: torch.LongTensor) -> BackwardOutput:
#         logits = torch.matmul(
#             input_embeds, 
#             self._net.pretrained_model.model.embed_tokens.weight.T
#         )
#         return BackwardOutput(
#             input_embeds=input_embeds,
#             logits=logits
#         )