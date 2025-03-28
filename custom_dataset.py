import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
class FinalDataset(Dataset):
    def __init__(self, file_path,indices=None):
        """
        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            indices (list, optional): List of indices to sample from the dataset.
                If None, all samples will be used. Defaults to None.
        """
        self.dataset = h5py.File(file_path, 'r')
        self.signals = self.dataset['signals']
        self.labels = self.dataset['labels'] 

        self.n_samples = len(self.signals)
        self.indices = indices if indices is not None else np.arange(self.n_samples)
        
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.indices)
    
    def __getitem__(self, index):
        signal = torch.tensor(self.signals[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        
        return signal, label
