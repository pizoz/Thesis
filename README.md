# Deep Learning for Chagas Disease Screening from ECGs

This repository contains the work from the thesis "**Deep Learning Models for Screening of Chagas Disease from Electrocardiograms**". The project explores the use of deep learning, specifically Residual Neural Networks (ResNets), to create an effective, low-cost, and non-invasive screening tool for Chagas disease using 12-lead ECG signals.

The main goal was to validate existing research and investigate the nature of the diagnostic problem: are the most important features found in the morphology of a single heartbeat, or in the temporal dynamics of the entire ECG recording?

## About The Project

This research focused on developing and interpreting deep learning models for a critical global health challenge. The work was divided into three key experimental paths:

* **Full ECG Analysis**: Training a ResNet model on complete ECG recordings to capture both beat morphology and temporal patterns.
* **Average Beat Analysis (QRS Complex)**: Focusing exclusively on the morphology of the average QRS complex to analyze ventricular depolarization.
* **Average Beat Analysis (QRST Complex)**: Expanding the analysis to include the T-wave, incorporating information about ventricular repolarization.

A key component of this project was the use of **Grad-CAM**, an Explainable AI (XAI) technique, to visualize and understand the models' decision-making processes. The resulting activation maps were compared with analyses from a medical expert to ensure the models were learning clinically relevant features.

### Key Findings

* The model trained on the **full ECG signal** achieved the best performance, suggesting that beat-to-beat variability and temporal dynamics are crucial for accurate diagnosis.
* Including the **T-wave** in the analysis significantly improved performance over using the QRS complex alone, highlighting the diagnostic value of the ventricular repolarization phase.
* **Grad-CAM** analysis confirmed that the models learned to identify clinically relevant markers, such as QRS abnormalities, T-wave alterations, and arrhythmias.

### Built With

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn

## Dataset

The project utilized data made available for the **George B. Moody Challenge 2025: "Detection of Chagas Disease from the ECG"**. This included three distinct cohorts:

* **CODE-15%**: A large dataset from Brazil with self-reported labels for Chagas disease.
* **SaMi-Trop**: A dataset from Brazil consisting of patients with serologically confirmed Chagas cardiomyopathy.
* **PTB-XL**: A large European dataset used as a negative control group.

The raw data underwent a rigorous preprocessing pipeline, including band-pass filtering, resampling to a common frequency (400 Hz), and truncation to a uniform length.

## Results

The final optimized model, trained on full ECG signals, demonstrated superior performance compared to both the baseline models from the literature and the models trained on average beats. The results on the test set, using the F1-Score maximizing threshold, are summarized below.

| Model                  | AUC-ROC | F1-Score | Precision | Recall |
| ---------------------- | ------- | -------- | --------- | ------ |
| **Full ECG (Final Model)** | **0.85**| **0.78** | **0.71** | **0.85** |
| Average Beat (QRST)    | 0.82    | 0.77     | 0.69      | 0.87   |
| Average Beat (QRS only)| 0.81    | 0.74     | 0.66      | 0.83   |

These findings strongly indicate that an approach analyzing the entire ECG trace is the most effective strategy for this classification task.
