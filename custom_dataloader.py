import h5py
import numpy as np
import torch

class Custom_Dataloader(object):
    
    def __init__(self,dataset,indices=None, batch_size=32, shuffle=False):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if indices is None:
            self.indices = np.arange(len(dataset))
        else:
            self.indices = indices
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        self.current_index = 0
        
    def __iter__(self):
        
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        
        if self.current_index >= len(self.indices):
            raise StopIteration
        
        start = self.current_index
        end = min(self.current_index + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        
        batch_signals = []
        batch_labels = []
        
        for i in batch_indices:
            signal, label = self.dataset[i]
            batch_signals.append(signal)
            batch_labels.append(label)
        
        batch_signals = torch.tensor(np.array(batch_signals))
        batch_labels = torch.tensor(np.array(batch_labels))
        
        self.current_index = end
        
        return batch_signals, batch_labels
    

    