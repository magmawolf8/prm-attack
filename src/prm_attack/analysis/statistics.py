"""Estimates cross-entropy and other statistics."""




# configuration
from prm_attack.config import DEVICE
# tensor modules
import torch
# python modules
from collections import Counter




class Statistics:
    def __init__(self, dataset, tokenizer_api, prm, device=DEVICE, batch_size=8):
        self.dataset = list(dataset)
        self.tokenizer_api = tokenizer_api
        self.prm = prm.to(device)
        self.device = device
        self.batch_size = batch_size

    def cross_entropy(self):
        """For binary classification tasks, the cross-entropy is
        CE = -1/N \sum{ y*log(p) + (1-y)*log(1-p) }
        """
        total_loss = 0.0
        total_steps = 0

        self.prm.eval()
        with torch.no_grad():
            for start in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset[start : start + self.batch_size]
                questions   = [ex["problem"] for ex in batch]
                steps_list  = [ex["steps"]   for ex in batch]
                labels      = [ex["label"]   for ex in batch]

                inputs = self.tokenizer_api.prepare_steps(questions, steps_list)
                inputs = inputs.to(self.device)

                out = self.prm(**inputs, return_prob=True)
                rewards = out.rewards         
                flags   = inputs["reward_flags"]   

                for prob_row, flag_row, label in zip(rewards, flags, labels):
                    p_steps = prob_row[flag_row.bool()]
                    n_steps = p_steps.numel()

                    if label == -1:
                        gt = torch.ones_like(p_steps, device=self.device)
                    else:
                        idx = int(label)
                        gt = torch.ones_like(p_steps, device=self.device)
                        if idx < n_steps:
                            gt[idx:] = 0.0

                    # clamping...
                    eps = 1e-12
                    p_clamped = torch.clamp(p_steps, eps, 1.0 - eps)

                    ce = -(gt * torch.log(p_clamped) + (1 - gt) * torch.log(1 - p_clamped))
                    total_loss += ce.sum().item()
                    total_steps += n_steps

        return total_loss / total_steps