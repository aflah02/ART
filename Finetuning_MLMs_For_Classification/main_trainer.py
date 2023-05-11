from model import MLMForClassification
from trainer_helpers import *
import pandas as pd
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_scheduler
from tqdm import tqdm, trange
from config import *
import wandb
import datetime

set_random_seed(SEED)

wandb.init(project="finetuning-mlms-for-classification")

# Build Dataset

train_dataset = pd.read_csv(TRAIN_DATA_FILE_PATH)
validation_dataset = pd.read_csv(VALIDATION_DATA_FILE_PATH)
test_dataset = pd.read_csv(TEST_DATA_FILE_PATH)

train_dataloader, validation_dataloader, test_dataloader = data_preproc_and_setup(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    test_dataset=test_dataset,
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
    padding_strategy=PADDING_STRATEGY,
    truncation=TRUNCATION,
)

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

wandb.watch(model)

early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)

for epoch in trange(NUM_EPOCHS):

    if LOG_STATUS_TO_TXT_FILE:
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Starting epoch {epoch} at {datetime.datetime.now()}")

    model.train()
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        expected_outputs = batch['label']
        loss = CrossEntropyLoss()(outputs, expected_outputs)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    if LOG_STATUS_TO_TXT_FILE:
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Finished epoch {epoch} at {datetime.datetime.now()}")
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Starting Model Save at {datetime.datetime.now()}")

    if SAVE_AFTER_EACH_EPOCH:
        torch.save(model.state_dict(), MODEL_CHECKPOINT_SAVE_DIR)

    if LOG_STATUS_TO_TXT_FILE:
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Finished Model Save at {datetime.datetime.now()}")
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Starting train evaluation for epoch {epoch} at {datetime.datetime.now()}")

    model.eval()
    perform_eval(model = model, split = 'train', dataloader = train_dataloader, device = DEVICE,
                out_dim=OUT_DIMENSION, save_dir=TRAIN_EVALUATION_RESULTS_SAVE_DIR,
                epoch=epoch, file_name = f"train_metrics", check_early_stopping=False,
                early_stopper=None, log_wandb=LOG_WANDB)

    if LOG_STATUS_TO_TXT_FILE:
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Finished train evaluation for epoch {epoch} at {datetime.datetime.now()}")
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Starting validation evaluation for epoch {epoch} at {datetime.datetime.now()}")

    model.eval()
    early_stop = perform_eval(model = model, split = 'validation', dataloader = validation_dataloader, device = DEVICE,
                out_dim=OUT_DIMENSION, save_dir=VALIDATION_EVALUATION_RESULTS_SAVE_DIR,
                epoch=epoch, file_name = f"validation_metrics", check_early_stopping=True,
                early_stopper=early_stopper, log_wandb=LOG_WANDB)
    
    if early_stop:
        if LOG_STATUS_TO_TXT_FILE:
            log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Early Stopping at epoch {epoch} at {datetime.datetime.now()}")
        break
   
    if LOG_STATUS_TO_TXT_FILE:
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Finished validation evaluation for epoch {epoch} at {datetime.datetime.now()}")
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Starting testing evaluation for epoch {epoch} at {datetime.datetime.now()}")

    model.eval()
    perform_eval(model = model, split = 'test', dataloader = validation_dataloader, device = DEVICE,
                out_dim=OUT_DIMENSION, save_dir=TEST_EVALUATION_RESULTS_SAVE_DIR,
                epoch=epoch, file_name = f"test_metrics", check_early_stopping=False,
                early_stopper=None, log_wandb=LOG_WANDB)

    if LOG_STATUS_TO_TXT_FILE:
        log_status_to_txt_file(LOG_TEXT_FILE_PATH, f"Finished testing evaluation for epoch {epoch} at {datetime.datetime.now()}")

        


            


        



