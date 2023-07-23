import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer
from custom_ds import CustomDataset
import wandb

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EarlyStopper:
    """
    Source: https://stackoverflow.com/a/73704579/13858953
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def compute_metrics(predictions, true_labels, split_name, epoch):
    """
    Helper function to compute metrics
    Args:
        predictions (list): list of predictions
        true_labels (list): list of true labels
    Returns:
        dict: dictionary of metrics
    """
    accuracy = accuracy_score(true_labels, predictions)
    macro_precision = precision_score(true_labels, predictions, average='macro')
    macro_recall = recall_score(true_labels, predictions, average='macro')
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    micro_precision = precision_score(true_labels, predictions, average='micro')
    micro_recall = recall_score(true_labels, predictions, average='micro')
    micro_f1 = f1_score(true_labels, predictions, average='micro')
    weighted_precision = precision_score(true_labels, predictions, average='weighted')
    weighted_recall = recall_score(true_labels, predictions, average='weighted')
    weighted_f1 = f1_score(true_labels, predictions, average='weighted')
    class_wise_precision = precision_score(true_labels, predictions, average=None)
    class_wise_recall = recall_score(true_labels, predictions, average=None)
    class_wise_f1 = f1_score(true_labels, predictions, average=None)
    return {
        'epoch': epoch,
        f'{split_name}_accuracy': accuracy,
        f'{split_name}_macro_precision': macro_precision,
        f'{split_name}_macro_recall': macro_recall,
        f'{split_name}_macro_f1': macro_f1,
        f'{split_name}_micro_precision': micro_precision,
        f'{split_name}_micro_recall': micro_recall,
        f'{split_name}_micro_f1': micro_f1,
        f'{split_name}_weighted_precision': weighted_precision,
        f'{split_name}_weighted_recall': weighted_recall,
        f'{split_name}_weighted_f1': weighted_f1,
        f'{split_name}_class_wise_precision': class_wise_precision,
        f'{split_name}_class_wise_recall': class_wise_recall,
        f'{split_name}_class_wise_f1': class_wise_f1
    }

def perform_eval(model, split, dataloader, device, out_dim, save_dir, epoch, file_name, check_early_stopping=False, early_stopper=None, log_wandb=True):
    model.eval()
    with torch.no_grad():
        predictions = []
        labels = []
        prediction_probabilities = []
        label_probabilities = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs, dim=1)
            expected_outputs = batch['label']
            # one hot encoded expected_outputs
            expected_outputs = torch.nn.functional.one_hot(expected_outputs, num_classes=out_dim)
            for prediction, expected_output in zip(predictions, expected_outputs):
                predictions.append(prediction.item())
                labels.append(expected_output.item())
            for prediction_probability, expected_output_probability in zip(outputs, expected_outputs):
                prediction_probabilities.append(prediction_probability)
                label_probabilities.append(expected_output_probability)
        metrics = compute_metrics(predictions, labels, split, epoch)
        loss = CrossEntropyLoss()(torch.stack(prediction_probabilities), torch.stack(label_probabilities))
        metrics['loss'] = loss.item()

        if log_wandb:
            wandb.log(metrics)

        # Save validation metrics
        with open(os.path.join(save_dir, f"{file_name}_epoch_{epoch}.json"), 'w') as f:
            json.dump(metrics, f)

        if check_early_stopping:
            bool_stop = early_stopper.early_stop(loss.item())
            if bool_stop:
                print("Early stopping")
                return True
    return False


def data_preproc_and_setup(train_dataset, validation_dataset, test_dataset, MODEL_NAME, PADDING_STRATEGY, TRUNCATION, BATCH_SIZE):
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

    return train_dataloader, validation_dataloader, test_dataloader, data_collator

def log_status_to_txt_file(save_path, status, log_bool):
    if not log_bool:
        return
    with open(save_path, 'a') as f:
        f.write(status + '\n')

if __name__ == "__main__":
    true_labels = [0, 1, 2, 0, 1, 2]
    predictions = [0, 2, 1, 0, 0, 1]
    metrics = compute_metrics(predictions, true_labels)
    print(metrics)
