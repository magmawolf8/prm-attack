SKYWORK_MODEL_NAME = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
#SKYWORK_MODEL_NAME = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
STEP_TOKEN = "\n\n"
RANDOM_SEED = 420

#********************************
#         Attack objective
#********************************
# Options: "flatten" (squared margin) or "invert" (flip discrimination)
ATTACK_OBJECTIVE = "invert"

#********************************
#      Continuous hyperparameters
#********************************
# General hyperparameters
NUM_PREFIXES = 30

# Training time hyperparameters
NUM_EPOCHS = 150
DATA_SUBSET_SIZE = 1
BATCH_SIZE = 1

# Optimizer behavior
LEARNING_RATE = 1
# Entropy regularization schedule
MIN_LAMBDA = 0.0
MAX_LAMBDA = 1e-2
T_MAX = 1.0
T_MIN = 1.0

#********************************
#        Discrete hyperparameters
#********************************
LM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_TOKEN_CANDIDATES = 80
MAX_NEW_TOKENS = 5
