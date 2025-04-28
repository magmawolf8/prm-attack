"""Model wrapper. Provides forward and backward pass on the PRM, and m-
anages to directly inject embeddings into the PRM's backbone LLM. Viol-
ates the abstraction barrier to set input embeddings and get the gradi-
ent with respect to the input embeddings. Also converts mixtures of di-
screte tokens and continuous vector embeddings into uniform vector emb-
eddings, and does the opposite too."""




# python modules
from dataclasses import dataclass
# tensor modules
from skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL
import torch




@dataclass
class ForwardOutput:
    input_embeds: torch.Tensor
    logits: torch.Tensor
    rewards: torch.Tensor


class _TransparentPRM(PRM_MODEL):

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)

    def forward(self, input_tensor: torch.LongTensor | torch.FloatTensor,
                      attention_mask: torch.BoolTensor,
                      skip_embedding_layer: bool = False,
                      return_prob: bool = False,
                      **kwargs) -> ForwardOutput:
        lm_output = None
        if skip_embedding_layer:
            lm_output = self.pretrained_model(
                input_embeds=input_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
        else:
            lm_output = self.pretrained_model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )

        lm_logits = lm_output.logits
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
            logits=lm_logits, 
            rewards=value, 
            input_embeds=lm_output.hidden_states[0]
        )


class PRMAPI:
    def __init__(self, model_name: str, device: str | torch.device = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device)
        self._net = _TransparentPRM.from_pretrained(model_name).to(self.device)
        self._net = self._net.eval()

    def forward_reward_cont(self, input_embeds: torch.Tensor,
                                  attention_mask: torch.BoolTensor,
                                  ) -> ForwardOutput:
        input_embeds = input_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self._net(
            input=input_embeds,
            attention_mask=attention_mask,
            skip_embedding_layer=True
        )

    def forward_reward(self, input_ids: torch.LongTensor,
                                 attention_mask: torch.BoolTensor,
                                 ) -> ForwardOutput:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self._net(
            input=input_ids,
            attention_mask=attention_mask
        )

    def forward_reward_prob(self, input_ids: torch.LongTensor,
                                  attention_mask: torch.BoolTensor,
                                  ) -> ForwardOutput:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self._net(
            input=input_ids,
            attention_mask=attention_mask,
            return_prob=True
        )

    def backward_gradients(self, rewards: torch.Tensor,
                                 input_embeds: torch.Tensor) -> torch.Tensor:
        """We could use the backward gradient function to do a little
        More logic regarding forward and backward passes, and pass back
        a BackwardOutput dataclass."""
        rewards.backward(
            gradient=torch.ones(rewards.shape).to(self.device), 
            inputs=input_embeds
        )
        return input_embeds.grad