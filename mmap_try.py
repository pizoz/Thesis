import h5py
import numpy as np

hdf5_file = "finalData.hdf5"

with h5py.File(hdf5_file,"r") as f:
    dataset_name = "signals"
    dataset = f[dataset_name]
    second_dataset = f["labels"]
    mmap_data = np.memmap(hdf5_file, mode="r", shape=dataset.shape, dtype=dataset.dtype)
    mmap_labels = np.memmap(hdf5_file, mode="r", shape=second_dataset.shape, dtype=second_dataset.dtype)

    # Example: Read a small portion without loading the entire file
    sample = mmap_data
    print("Sample data:", sample.shape)
    print("Sample labels:", mmap_labels.shape)
    
    
