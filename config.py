SKYWORK_MODEL_NAME = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
STEP_TOKEN = "\n\n"
RANDOM_SEED = 420

#********************************
#      Continuous hyperparameters
#********************************
# General hyperparameters
NUM_PREFIXES = 30

# Training time hyperparameters
NUM_EPOCHS = 200
DATA_SUBSET_SIZE = 1
BATCH_SIZE = 1

# Optimizer behavior
LEARNING_RATE = 0.5
# Entropy regularization schedule
MIN_LAMBDA = 0.001
MAX_LAMBDA = 0.1
TAU = 0.5

#********************************
#        Discrete hyperparameters
#********************************
LM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_TOKEN_CANDIDATES = 80
MAX_NEW_TOKENS = 5