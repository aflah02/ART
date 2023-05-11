import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        return {
            'input_ids': self.ds[index]['input_ids'],
            'attention_mask': self.ds[index]['attention_mask'],
            'label': self.ds[index]['label']
        }

    def __len__(self):
        return len(self.ds)