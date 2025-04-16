from custom_dataset import FinalDataset

dataset1 = FinalDataset("dataset.hdf5")
dataset2 = FinalDataset("dataset.hdf5", downsample=True, majority_ratio=0.95)
print(f"Dataset 1 length: {len(dataset1)}")
print(f"Dataset 2 length: {len(dataset2)}")