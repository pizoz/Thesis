import torch
import h5py
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.data = f['signals'][:10000]  # Load subset
            self.labels = f['labels'][:10000]  # Load subset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)  # Convert to tensor
        y = torch.tensor(self.labels[idx], dtype=torch.long)   # Convert to tensor
        return x, y