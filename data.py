import os
import csv
import h5py
import numpy as np
import wfdb
import pandas as pd
from wfdb.processing import resample_sig
from scipy.signal import butter, filtfilt

# Funzione per il filtro passa banda
def butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=400, order=3):
    nyquist_freq = 0.5 * fs
    low, high = lowcut / nyquist_freq, highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Funzione per ridurre la lunghezza del segnale
def shorten_signal(signal, target_size=4000):
    return signal[:target_size] if signal.shape[0] > target_size else signal

# Funzione per il resampling
def resample(signal, fs, fs_new):
    resampled_signal = np.zeros((4000, signal.shape[1]))  # Allocazione fissa per evitare problemi
    for i in range(signal.shape[1]):
        res_sig, _ = resample_sig(signal[:, i], fs, fs_new)
        resampled_signal[:, i] = res_sig
    return resampled_signal

# Lettura delle etichette dal CSV
exam_labels = {}
with open("data/merged.csv", newline="\n") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        exam_labels[int(row["exam_id"])] = bool(row["chagas"])

# Preparazione del file HDF5 con `np.memmap` per evitare il sovraccarico di RAM
output_file = "finalData.hdf5"
signals_memmap = np.memmap("finalData_signals.dat", dtype="float32", mode="w+", shape=(369209, 4000, 12))
labels_memmap = np.memmap("finalData_labels.dat", dtype="int32", mode="w+", shape=(369209,))

# Lettura dei file HDF5 con segnali ECG
files = [f"./data/exams_part{i}.hdf5" for i in range(18)]
index = 0

for filename in files:
    if not os.path.exists(filename):
        continue  # Skip file se non esiste
    
    with h5py.File(filename, "r") as ecgs:
        exam_ids = list(ecgs["exam_id"])
        for i, exam_id in enumerate(exam_ids):
            if exam_id not in exam_labels:
                continue  # Skip se non è in exam_labels
            
            signal = butter_bandpass_filter(ecgs["tracings"][i])
            signal = shorten_signal(signal)

            #check if the signal is not empty
            if np.all(signal == 0):
                continue
            signals_memmap[index] = signal
            labels_memmap[index] = int(exam_labels[exam_id])
            index += 1
    print(f"Processed {filename}")
# Lettura del dataset SAMI
with h5py.File("data/examsSAMI.hdf5", "r") as ecgs:
    sami_exam_ids = pd.read_csv("data/examsSAMI.csv")["exam_id"].tolist()
    for i, exam_id in enumerate(sami_exam_ids):
        signal = butter_bandpass_filter(ecgs["tracings"][i])
        signal = shorten_signal(signal)

        # Scrivo direttamente su `memmap`
        if np.all(signal == 0):
            continue
        signals_memmap[index] = signal
        labels_memmap[index] = 1  # Tutti i SAMI sono `True`
        index += 1
print(f"Processed SAMI")
# Lettura del dataset PTB-XL
ptb_labels = pd.read_csv("data/ptbxl_database.csv")[["ecg_id"]]
ptb_labels["label"] = False  # Tutti i PTB sono `False`
ptb_records = "data/records500/"

for subfolder in sorted(os.listdir(ptb_records)):
    subfolder_path = os.path.join(ptb_records, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    for dat_file in [f for f in os.listdir(subfolder_path) if f.endswith(".dat")]:
        record_name = os.path.join(subfolder_path, dat_file[:-4])
        try:
            record = wfdb.rdsamp(record_name)
            signals, _ = record
            signals = butter_bandpass_filter(signals)
            signals = resample(signals, 500, 400)
            signals = shorten_signal(signals)

            # Scrivo direttamente su `memmap`
            if np.all(signal == 0):
                continue
            signals_memmap[index] = signals
            labels_memmap[index] = 0  # PTB-XL è `False`
            index += 1
        except Exception as e:
            print(f"Errore su {dat_file}: {e}")
print(f"Processed PTB-XL")
# Salvataggio nel file HDF5
with h5py.File(output_file, "w") as f:
    f.create_dataset("signals", data=signals_memmap[:index])
    f.create_dataset("labels", data=labels_memmap[:index])

# Eliminazione dei file temporanei memmap
del signals_memmap, labels_memmap
os.remove("finalData_signals.dat")
os.remove("finalData_labels.dat")

print(f"✅ Dataset salvato in {output_file} con {index} segnali!")
