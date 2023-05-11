from model import MLMForClassification
from trainer_helpers import *
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer
from torch.nn import CrossEntropyLoss
from custom_ds import CustomDataset
from transformers import AdamW, get_scheduler
from tqdm import tqdm, trange

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

# Neural Network Architecture
LS_ACTIVATIONS = [nn.ReLU(), nn.ReLU()]
SOFTMAX_OUTPUTS = True
FC_DIMENSIONS = [100, 50]
OUT_DIMENSION = 2
LS_LAYERS_TO_FREEZE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
LS_ONLY_FREEZE_PARAMETERS_HAVING_THIS_SUBSTRING = []
FN_TO_PASS_HIDDEN_STATE_THROUGH_BEFORE_CLASSIFICATION_HEAD = None
USE_BACKBONE_FROM_CACHE = True

set_random_seed(SEED)

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
VALIDATION_EVALUATION_RESULTS_SAVE_DIR = r""
TESTING_EVALUATION_RESULTS_SAVE_DIR = r""

# Build Dataset

train_dataset = pd.read_csv(TRAIN_DATA_FILE_PATH)
validation_dataset = pd.read_csv(VALIDATION_DATA_FILE_PATH)
test_dataset = pd.read_csv(TEST_DATA_FILE_PATH)

# Only keep the text and label columns
train_dataset = train_dataset[['text', 'label']]
validation_dataset = validation_dataset[['text', 'label']]
test_dataset = test_dataset[['text', 'label']]

# Build Dataloaders
dataset_hf = DatasetDict({
    'train': Dataset.from_pandas(train_dataset),
    'test': Dataset.from_pandas(test_dataset),
    'validation': Dataset.from_pandas(validation_dataset)
})

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset_hf = dataset_hf.map(lambda x: tokenizer(x['text'], 
                                                padding=PADDING_STRATEGY, 
                                                truncation=TRUNCATION), 
                                                batched=True, 
                                                batch_size=BATCH_SIZE)
dataset_hf.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Build DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataset = dataset_hf['train']
validation_dataset = dataset_hf['validation']
test_dataset = dataset_hf['test']

train_dataset = CustomDataset(train_dataset)
validation_dataset = CustomDataset(validation_dataset)
test_dataset = CustomDataset(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Build Model
model = MLMForClassification(hub_model_name=MODEL_NAME,
                            ls_activations=LS_ACTIVATIONS,
                            softmax_outputs=SOFTMAX_OUTPUTS,
                            fc_dimensions=FC_DIMENSIONS,
                            out_dimension=OUT_DIMENSION,
                            ls_layers_to_freeze=LS_LAYERS_TO_FREEZE,
                            ls_only_freeze_parameters_having_this_substring=LS_ONLY_FREEZE_PARAMETERS_HAVING_THIS_SUBSTRING,
                            fn_to_pass_hidden_state_through_before_classification_head=FN_TO_PASS_HIDDEN_STATE_THROUGH_BEFORE_CLASSIFICATION_HEAD,
                            use_backbone_from_cache=USE_BACKBONE_FROM_CACHE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = get_scheduler(
                    'linear',
                    optimizer = optimizer,
                    num_warmup_steps = NUM_WARMUP_STEPS,
                    num_training_steps = NUM_TRAINING_STEPS
                )

# for epoch in trange(NUM_EPOCHS):
#     model.train()
#     for batch in tqdm(train_dataloader):
#         batch = {k: v.to(DEVICE) for k, v in batch.items()}
#         outputs = model(**batch)
#         expected_outputs = batch['label']
#         loss = CrossEntropyLoss()(outputs, expected_outputs)
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm(validation_dataloader):
#             batch = {k: v.to(DEVICE) for k, v in batch.items()}
#             outputs = model(**batch)
#             predictions = torch.argmax(outputs, dim=1)
#             expected_outputs = batch['label']
#             accuracy = (predictions == expected_outputs).float().mean()


        



