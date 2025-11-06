SKYWORK_MODEL_NAME = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
STEP_TOKEN = "\n\n"
RANDOM_SEED = 420

#********************************
#      Continuous hyperparameters
#********************************
# General hyperparameters
NUM_PREFIXES = 1

# Training time hyperparameters
NUM_EPOCHS = 1
DATA_SUBSET_SIZE = 1000
BATCH_SIZE = 1

# Optimizer behavior
LEARNING_RATE = 5e-6
REG_LAMBDA = 0.0
MOMENTUM = 0.0

#********************************
#        Discrete hyperparameters
#********************************
LM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_TOKEN_CANDIDATES = 80
MAX_NEW_TOKENS = 5