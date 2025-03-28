from custom_dataset import FinalDataset
from custom_Dataloader import CustomDataloader

file_path = 'D:\\thesis_data\\filteredData.hdf5'

dataset = FinalDataset(file_path)

dataloader = CustomDataloader(dataset, batch_size=32, shuffle=True)
for batch_signals, batch_labels in dataloader:
    print("Batch Signals Shape:", batch_signals.shape)
    print("Batch Labels Shape:", batch_labels.shape)