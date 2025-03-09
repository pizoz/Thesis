import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import h5py
class FinalDataset(Dataset):
    
    def __init__(self, file_path):
        """
        Args:
            open_file (file): open hdf5 file with the data. Make tuples and save them in a list of tuples.
        """
        with h5py.File(file_path, 'r') as open_file:
            self.signals = open_file['signals'][:]
            self.labels = open_file['labels'][:]
        
        self.data = list(zip(self.signals, self.labels))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal, label = self.data[idx]

        return signal, label
    
    def plotSignal(self, idx):
        signal, label = self[idx]

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
