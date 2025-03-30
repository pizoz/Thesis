import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
class FinalDataset(Dataset):
    def __init__(self, file_path,indices=None):
        """
        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            indices (list, optional): List of indices to sample from the dataset.
                If None, all samples will be used. Defaults to None.
                This is useful for sampling a subset of the dataset for training or evaluation.
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
    
    def plotSignal(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        num_leads = signal.shape[1]

        fig, axes = plt.subplots(num_leads, 1, figsize=(10, 8), sharex=True)

        lead_names = [
            "Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"
        ]
        colors = ["black", "red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", "magenta"]
        
        for i in range(num_leads):
            axes[i].plot(signal[:, i], color=colors[i])
            axes[i].set_ylabel(lead_names[i], fontsize=8, rotation=0, labelpad=30)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        
        axes[-1].set_xlabel("Time")
        title = "Signal without Chagas" if label == 0 else "Signal with Chagas"
        fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        plt.show()
