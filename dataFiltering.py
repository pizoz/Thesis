import os
import h5py
import numpy as np

# File di input e output
input_file = "finalData.hdf5"
temp_signals_file = "filtered_signals.dat"
temp_labels_file = "filtered_labels.dat"
output_file = "D:/filteredData.hdf5"

# Apertura sicura del file HDF5 originale
with h5py.File(input_file, "r") as f:
    signals = f["signals"]
    labels = f["labels"]
    total_samples = signals.shape[0]

    # Creazione di memmap temporanei per i dati filtrati
    filtered_signals = np.memmap(temp_signals_file, dtype="float32", mode="w+", shape=(total_samples, 4000, 12))
    filtered_labels = np.memmap(temp_labels_file, dtype="int32", mode="w+", shape=(total_samples,))

    index = 0
    for i in range(total_samples):
        signal = signals[i, :, :] 

        if not np.all(signal == 0):
            filtered_signals[index] = signal
            filtered_labels[index] = labels[i] 
            index += 1
            print(index, end="\r")

# Scrittura nel nuovo file HDF5
with h5py.File(output_file, "w") as f:
    f.create_dataset("signals", data=filtered_signals[:index])
    f.create_dataset("labels", data=filtered_labels[:index])

# Rimozione dei file temporanei
del filtered_signals, filtered_labels
os.remove(temp_signals_file)
os.remove(temp_labels_file)

print(f"✅ Dataset filtrato salvato in {output_file} con {index} segnali validi!")
