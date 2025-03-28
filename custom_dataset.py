import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class FinalDataset(Dataset):
    def __init__(self, file_path,indices=None):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            indices (list, optional): List of indices to sample from the dataset. If None, all indices are used.
        """
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.lenght = len(f['signals'])
        self.indices = indices if indices is not None else np.arange(self.lenght)
    
    def __len__(self):
        return self.lenght

    def __getitem__(self, idx):
        
        with h5py.File(self.file_path, 'r') as f:
            signal = f['signals'][idx]
            label = f['labels'][idx]
        
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return signal, label

    def plotSignal(self, idx):
        
        with h5py.File(self.file_path, 'r') as f:
            signal = f['signals'][idx]
            label = f['labels'][idx]
        
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


        