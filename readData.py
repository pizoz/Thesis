import h5py
import csv
import numpy as np
import os
import pandas as pd

# HDF5 file with exam_id and tracings

filename = "./data/exams_part0.hdf5"

# CSV file with exam_id, patient_Id and label
labels_filename = "./data/code15_chagas_labels.csv"

df_sample1 = pd.read_csv(labels_filename)

# main CSV file
main_filename = "./data/exams.csv"

df_sample2 = pd.read_csv(main_filename)

lista = ['exam_id','patient_id']

# csv file with exam_id, patient_id and label
df_master = pd.merge(df_sample2, df_sample1, on=lista, how='left')

# saving it as  a file

df_master.to_csv('data/merged.csv', index=False)

# reading file HDF5 separing the two datasets
ecgs = h5py.File(filename, 'r')
exam_ids = np.array(ecgs['exam_id'])
tracings = np.array(ecgs['tracings'])

# pupulating the dictionary exam_ids_and_signals
exam_ids_to_chagas = dict()

with open("data/merged.csv", newline="\n") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        exam_id = int(row['exam_id'])
        boolean = bool(row['chagas'])
        exam_ids_to_chagas[exam_id] = boolean

# list of files to read
files = ["./data/exams_part"+str(i)+".hdf5" for i in range(0,18)]

# populating the dictionary exam_ids_and_signals
exam_ids_and_signals = dict()
for filename in files:
    if (os.path.exists(filename) == False):
        continue
    print("Reading file: ", filename)
    with h5py.File(filename, "r") as ecgs:

        exam_ids = list(ecgs['exam_id'])

        num_exams = len(exam_ids)

        for i in range(num_exams):
            exam_id = exam_ids[i]

            if exam_id not in exam_ids_to_chagas:
                continue
            else:
                exam_ids_and_signals[exam_id] = ecgs['tracings'][i]

# so at the end i got
# exam_ids_to_chagas that maps the exam id to Chagas disease label
# exam_ids_and_signals that maps the exam id to the ECG signal

print("Number of ECGs: ", len(exam_ids_and_signals))