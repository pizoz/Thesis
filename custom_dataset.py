import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class FinalDataset(Dataset):
    def __init__(self, file_path, indices=None, downsample=False, majority_ratio=0.95):
        """
        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            indices (list, optional): Specific indices to use from the dataset.
            downsample (bool): Whether to downsample the majority class. Defaults to False.
            majority_ratio (float): Ratio of majority class (e.g., 0.95 means 95% majority, 5% minority).
        """
        self.dataset = h5py.File(file_path, 'r')
        self.signals = self.dataset['signals']
        self.labels = self.dataset['labels']

        all_indices = np.arange(len(self.signals)) if indices is None else np.array(indices)

        if downsample:
            self.indices = self._controlled_downsample(all_indices, majority_ratio)
        else:
            self.indices = all_indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        real_index = self.indices[index]
        signal = torch.tensor(self.signals[real_index], dtype=torch.float32)
        label = torch.tensor(self.labels[real_index], dtype=torch.float32)
        return signal, label

    def get_labels(self):
        return self.labels[np.sort(self.indices)]

    def _controlled_downsample(self, indices, majority_ratio):
        
        labels = np.array(self.labels[indices]).squeeze()
        class_0_indices = indices[labels == 0]
        class_1_indices = indices[labels == 1]

        # Identify majority and minority
        if len(class_0_indices) > len(class_1_indices):
            majority_indices = class_0_indices
            minority_indices = class_1_indices
        else:
            majority_indices = class_1_indices
            minority_indices = class_0_indices

        n_minority = len(minority_indices)
        total_desired = int(n_minority / (1 - majority_ratio))
        n_majority = total_desired - n_minority
        n_majority = min(n_majority, len(majority_indices))  # Just in case

        majority_sampled = np.random.choice(majority_indices, size=n_majority, replace=False)
        combined = np.concatenate([majority_sampled, minority_indices])
        np.random.shuffle(combined)
        return combined

    def plotSignal(self, idx):
        real_index = self.indices[idx]
        signal = self.signals[real_index]
        label = self.labels[real_index]
        num_leads = signal.shape[1]

        fig, axes = plt.subplots(num_leads, 1, figsize=(10, 8), sharex=True)

        lead_names = [
            "Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"
        ]
        colors = [
            "black", "red", "blue", "green", "orange", "purple",
            "brown", "pink", "gray", "olive", "cyan", "magenta"
        ]
        
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

class BeatsDataset(Dataset):
    def __init__(self, file_path, indices=None, downsample=False, majority_ratio=0.95):
        """
        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            indices (list, optional): Specific indices to use from the dataset.
            downsample (bool): Whether to downsample the majority class. Defaults to False.
            majority_ratio (float): Ratio of majority class (e.g., 0.95 means 95% majority, 5% minority).
        """
        self.dataset = h5py.File(file_path, 'r')
        self.beats = self.dataset['beats']
        self.labels = self.dataset['labels']

        all_indices = np.arange(len(self.beats)) if indices is None else np.array(indices)

        if downsample:
            self.indices = self._controlled_downsample(all_indices, majority_ratio)
        else:
            self.indices = all_indices
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        real_index = self.indices[index]
        signal = torch.tensor(self.beats[real_index], dtype=torch.float32)
        label = torch.tensor(self.labels[real_index], dtype=torch.float32)
        return signal, label

    def get_labels(self):
        return self.labels[np.sort(self.indices)]
    
    def _controlled_downsample(self, indices, majority_ratio):
        
        labels = np.array(self.labels[indices]).squeeze()
        class_0_indices = indices[labels == 0]
        class_1_indices = indices[labels == 1]

        # Identify majority and minority
        if len(class_0_indices) > len(class_1_indices):
            majority_indices = class_0_indices
            minority_indices = class_1_indices
        else:
            majority_indices = class_1_indices
            minority_indices = class_0_indices

        n_minority = len(minority_indices)
        total_desired = int(n_minority / (1 - majority_ratio))
        n_majority = total_desired - n_minority
        n_majority = min(n_majority, len(majority_indices))  # Just in case

        majority_sampled = np.random.choice(majority_indices, size=n_majority, replace=False)
        combined = np.concatenate([majority_sampled, minority_indices])
        np.random.shuffle(combined)
        return combined