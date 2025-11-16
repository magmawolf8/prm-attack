"""Configuration for continuous adversarial PRM attack."""

from dataclasses import dataclass

# ======================================
# global seed
# ======================================

RANDOM_SEED = 420

# ======================================
# hyperparameters
# ======================================

@dataclass
class Hyperparameters:
    """Dataclass to hold all experiment hyperparameters."""
    
    # model and tokenizer
    SKYWORK_MODEL_NAME: str = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    STEP_TOKEN: str = "\n\n"

    # experiment control
    # If True, each epoch uses a full-batch.
    # If False, use stochastic mini-batches.
    FULL_BATCH: bool = True

    NUM_PREFIXES: int = 30

    # training schedule
    NUM_EPOCHS: int = 200
    DATA_SUBSET_SIZE: int = 1
    BATCH_SIZE: int = 1  # mini-batch size per GPU when FULL_BATCH=False
    # --- Entropy Regularization ---
    MIN_LAMBDA: float = 0.001  # start
    MAX_LAMBDA: float = 0.1   # end

    LEARNING_RATE: float = 0.5


    TAU: float = 0.5