import h5py
import numpy as np

class CustomDataloader:
    
    def __init__(self,file_path, batch_size=32, shuffle=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.file = h5py.File(self.file_path, 'r')
        self.signals = self.file['signals']
        self.labels = self.file['labels']
        self.total_samples = self.signals.shape[0]
        self.indices = np.arange(self.total_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        self.current_index = 0
        
    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current_index >= self.total_samples:
            self.file.close()
            raise StopIteration
        
        start = self.current_index
        end = min(self.current_index + self.batch_size, self.total_samples)
        batch_indices = self.indices[start:end]
        
        batch_signals = np.array([self.signals[i] for i in batch_indices])
        batch_labels = np.array([self.labels[i] for i in batch_indices])
        
        self.current_index = end
        
        return batch_signals, batch_labels
    