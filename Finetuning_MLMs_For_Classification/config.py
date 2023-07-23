import torch.nn as nn
import torch

# All Paths
"""
The input files should be in the following format:
    - The first column should be the text named "text"
    - The second column should be the label named "label"
    - All other columns will be ignored
    - The first row should be the header
    - The file should be a csv file
    - The labels should be preprocessed to be integers starting from 0
"""

TRAIN_DATA_FILE_PATH = r""
TEST_DATA_FILE_PATH = r""
VALIDATION_DATA_FILE_PATH = r""

MODEL_CHECKPOINT_SAVE_DIR = r""
TRAIN_EVALUATION_RESULTS_SAVE_DIR = r""
VALIDATION_EVALUATION_RESULTS_SAVE_DIR = r""
TEST_EVALUATION_RESULTS_SAVE_DIR = r""

# All Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
NUM_WARMUP_STEPS = 0
NUM_TRAINING_STEPS = 10000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'vinai/bertweet-base'
PADDING_STRATEGY = 'max_length'
TRUNCATION = True
SEED = 42
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0
LOG_WANDB = True
SAVE_AFTER_EACH_EPOCH = True
LOG_STATUS_TO_TXT_FILE = True
LOG_TEXT_FILE_PATH = r""

# Neural Network Architecture
LS_ACTIVATIONS = [nn.ReLU(), nn.ReLU()]
SOFTMAX_OUTPUTS = True
FC_DIMENSIONS = [100, 50]
OUT_DIMENSION = 2
LS_LAYERS_TO_FREEZE = [0, 1, 2, 3, 4, 5, 6, 7, 8]
LS_ONLY_FREEZE_PARAMETERS_HAVING_THIS_SUBSTRING = []
FN_TO_PASS_HIDDEN_STATE_THROUGH_BEFORE_CLASSIFICATION_HEAD = None
USE_BACKBONE_FROM_CACHE = True