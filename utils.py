
# ===============================
# custom datasets
# ===============================

class PRM800k(Dataset):
    """Dataset which serves the PRM800k dataset from a local .jsonl file."""

    def __init__(self, jsonl_path, size):
        self.samples = []
        print(f"Loading {size} samples from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            for idx, line in enumerate(f):
                if idx == size:
                    break
                if line.strip():
                    data = json.loads(line)
                    question = data["question"]["problem"]
                    answer = data["question"]["pre_generated_steps"]
                    self.samples.append((question, answer))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, answer = self.samples[idx]
        return question, answer


class SingleQADataset(Dataset):
    """Dataset which always returns the same (question, answer) pair."""

    def __init__(self, question, answer_steps, size=1):
        self.question = question
        self.answer_steps = answer_steps
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.question, self.answer_steps


# ===============================
# gradient sign optimizer
# ===============================

class FGSM(torch.optim.SGD):
    """Simple optimizer that steps in the sign of the gradient."""

    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr=lr, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad.sign(), alpha=-lr)

        return loss


# ===============================
# helper functions
# ===============================

def collate_into_batch(samples_list):
    questions, answers = zip(*samples_list)
    return list(questions), list(answers)