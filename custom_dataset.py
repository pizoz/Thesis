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
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the HDF5 file.
        """
        self.file_path = file_path
        
        with h5py.File(file_path, 'r') as f:
            self.data_shape = f['signals'].shape
            self.label_shape = f['labels'].shape
            self.data_dtype = f['signals'].dtype
            self.label_dtype = f['labels'].dtype
        
        # Memory mapping
        self.signals = np.memmap(file_path, mode='r', shape=self.data_shape, dtype=self.data_dtype)
        self.labels = np.memmap(file_path, mode='r', shape=self.label_shape, dtype=self.label_dtype)
    
    def __len__(self):
        return self.data_shape[0]

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
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
