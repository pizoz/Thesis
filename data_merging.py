import csv
import numpy as np
import h5py
import os
import torch
import scipy.signal
from wfdb.processing import resample_sig
import wfdb

labels_dictionary = dict()
file_path = "Dataset.hdf5"
temporary_signal_path = "D:\\signals.dat"
temporary_label_path = "D:\\labels.dat"
positives = 0
negatives = 0

"""
    Popolamento dizionari per coontare il numero di esami di cui ho le etichette
"""

"""
    Code15%
"""
with open("merged_file.csv", "r") as f:
    reader = csv.DictReader(f)
    header = next(reader)
    for row in reader:
        exam_id = row["exam_id"]
        label = row["chagas"]
        if label == "True":
            label = 1
            positives += 1
        else:
            label = 0
            negatives += 1
        labels_dictionary[exam_id] = label

positives_count = 0
for exam_id in labels_dictionary:
    if labels_dictionary[exam_id] == 1:
        positives_count += 1
print(f"Esami di CODE15% con etichetta positiva: {positives_count}")
print(f"Esami di CODE15%: {len(labels_dictionary)}")     

"""
    Di Sami Trop ho solo metà degli esami, quindi devo prendere solo quelli quando carico il dataset
"""
total_labels = 0
with h5py.File("exams.hdf5", 'r') as f:
    total_labels = len(f['tracings'])
    positives += total_labels
    
print(f"Esami di Sami Trop: {total_labels}")
"""
    PTB va aggiunto direttamente al file

"""  
n_ptb_xl_signals = 0
main_directory = "records500"

for subfolder in sorted(os.listdir(main_directory)):
    subfolder_path = os.path.join(main_directory, subfolder)
    if os.path.isdir(subfolder_path):
        
        dat_files = [f for f in os.listdir(subfolder_path) if f.endswith(".dat")]
        n_ptb_xl_signals += len(dat_files)
        negatives += len(dat_files)

print(f"Number of PTB XL signals: {n_ptb_xl_signals}")
print(f"Number of CODE15%: {len(labels_dictionary)}")
print(f"Number of Sami Trop: {total_labels}")
print(f"Total number of signals: {n_ptb_xl_signals + len(labels_dictionary)+total_labels}")
print(f"Total number of positives: {positives}")
print(f"Total number of negatives: {negatives}")
print(f"Total number of signals: {positives + negatives}")

# """
#     Now I have to merge all the dataset
    
# """

# def butter_bandpass_filter(data,lowcut, highcut, fs,order):

#     nyquist_freq = 0.5 * fs
#     low = lowcut / nyquist_freq
#     high = highcut / nyquist_freq

#     b,a = scipy.signal.butter(order, [low, high], btype='band')
#     y = scipy.signal.filtfilt(b, a, data,axis=0)

#     return y

# def shorten_Code(signal):
    
#     if signal.shape[0] > 4000:
#         # riconverto il segnale in un tensore, utilizzo la copy per evitare errori con stripe negativa
#         signal = torch.tensor(signal.copy())
#         # taglio il segnale a 10 secondi che corrispondono a 4000 campioni a 400 Hz
#         signal = torch.narrow(signal, 0, 0, 4000)
#     return signal

# def shorten_Sami(signal):
    
#     if signal.shape[0] == 4096:
        
#         start = 48
#         end = 4048
#         sliced_signal = signal[start:end,:]
#     return sliced_signal

# def resample(signal, fs, fs_new):
#     try:
#         resampled_signal = np.zeros((4000, signal.shape[1]))
#         for i in range(signal.shape[1]):

#             res_sig,_ = resample_sig(signal[:,i], fs, fs_new)
#             resampled_signal[:,i] = res_sig
#     except Exception as e:
#         print(e)
    
#     return resampled_signal

# signals_mmemmap = np.memmap(temporary_signal_path, dtype='float32', mode='w+', shape=(len(labels_dictionary)+total_labels+n_ptb_xl_signals, 4000, 12))
# labels_mmemmap = np.memmap(temporary_label_path, dtype='float32', mode='w+', shape=(len(labels_dictionary)+total_labels+n_ptb_xl_signals, 1))

# """
#     Code15%
# """

# files = ["exams_part"+str(i)+".hdf5" for i in range(0,18)]
# current_id = 0
# for file in files:
#     print(f"Loading {file}")
#     with h5py.File(file, 'r') as f:
#         signals = f['tracings']
#         exam_ids = f['exam_id']
        
#         for i in range(len(signals)):
#             exam_id = str(exam_ids[i])
#             if exam_id in labels_dictionary:
#                 signal = signals[i]
#                 label = labels_dictionary[exam_id]
#                 # filtro il segnale
#                 signal = butter_bandpass_filter(signal, 0.5, 40, 400, 3)
#                 # accorcio il segnale a 10 secondi
#                 signal = shorten_Code(signal)
#                 # metto il segnale in memoria
#                 signals_mmemmap[current_id] = signal.numpy()
#                 labels_mmemmap[current_id] = label
#                 current_id += 1
# """
#     Sami Trop
# """
                
# with h5py.File("exams.hdf5", 'r') as f:
#     print("Loading Sami Trop")
#     signals = f['tracings']
    
#     for i in range(len(signals)):
#         signal = signals[i]
#         label = 1
#         # filtro il segnale
#         signal = butter_bandpass_filter(signal, 0.5, 40, 400, 3)
#         # accorcio il segnale a 10 secondi
#         signal = shorten_Sami(signal)
#         # metto il segnale in memoria
#         signals_mmemmap[current_id] = signal
#         labels_mmemmap[current_id] = label
#         current_id += 1
            
# """
#     PTB XL
# """
# for subfolder in sorted(os.listdir(main_directory)):
#     print(f"Loading {subfolder}")
#     subfolder_path = os.path.join(main_directory, subfolder)
#     if os.path.isdir(subfolder_path):
        
#         dat_files = [f for f in os.listdir(subfolder_path) if f.endswith(".dat")]
#         for file in dat_files:
#             record = wfdb.rdsamp(os.path.join(subfolder_path, file[:-4]))
#             signals, fields = record
            
#             signals = butter_bandpass_filter(signals, 0.5, 40, 400, 3)
#             signals = resample(signals, 500, 400)
#             signals = shorten_Code(signals)
            
#             signals_mmemmap[current_id] = signals
#             labels_mmemmap[current_id] = 0
#             current_id += 1
            
# with h5py.File(file_path, 'w') as f:
#     f.create_dataset('signals', data=signals_mmemmap)
#     f.create_dataset('labels', data=labels_mmemmap)
            

