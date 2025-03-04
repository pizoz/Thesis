import torch
from torch.utils.data import Dataset

class FinalDataset(Dataset):
    
    def __init__(self, data):
        """
        Args:
            data (list of tuples): Each tuple is (signal, label).
        """
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal, label = self.data[idx]
        return signal, label
