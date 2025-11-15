"""Configuration for continuous adversarial PRM attack."""

# ======================================
# model + tokenizer
# ======================================

SKYWORK_MODEL_NAME = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
STEP_TOKEN = "\n\n"

# ======================================
# experiment control
# ======================================

RANDOM_SEED = 420

# If True, each epoch uses a single full-batch step per GPU
# (one optimizer step per epoch, progress bar over epochs).
# If False, use stochastic mini-batches with per-epoch progress bars.
FULL_BATCH = True

# ======================================
# continuous attack hyperparameters
# ======================================

# prefix parameterization
NUM_PREFIXES = 30

# training schedule
NUM_EPOCHS = 200
DATA_SUBSET_SIZE = 1
BATCH_SIZE = 1     # mini-batch size per GPU when FULL_BATCH=False

# optimizer behavior
LEARNING_RATE = 0.5

# entropy regularization
MIN_LAMBDA = 0.001 # start
MAX_LAMBDA = 0.1   # end

# Gumbel-softmax temperature
TAU = 0.5