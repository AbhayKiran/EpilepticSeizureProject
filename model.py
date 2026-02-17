# 01_cnn_only.py  (MULTI-PATIENT 5-MIN PREDICTION)

import os
import random
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, GlobalAveragePooling2D, Dense, Concatenate
)
from tensorflow.keras.layers import (
    MultiHeadAttention, LayerNormalization, Add,
    Reshape, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Embedding, Flatten

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import glob


class EEGSeizurePredictionCNN:
    def __init__(self, data_path, summaries_dir=None, preictal_sec=300,
                 max_non_seizure_files=10, random_seed=42):

        self.data_path = data_path
        self.summaries_dir = summaries_dir if summaries_dir else data_path
        self.preictal_horizon = preictal_sec
        self.max_non_seizure_files = max_non_seizure_files
        self.random_seed = random_seed

        self.cnn_model = None

        self.overlap = 0.25
        self.sampling_rate = 256
        self.window_size = 10
        self.n_channels = 23

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    # ======================================================
    # SUMMARY PARSER
    # ======================================================
    def _parse_all_summaries(self):

        file_info = {}

        summary_paths = glob.glob(
            os.path.join(self.data_path, "chb*", "*-summary.txt")
        )

        if not summary_paths:
            raise FileNotFoundError("No summary files found inside patient folders!")

        for sfile in summary_paths:
            with open(sfile, 'r') as f:
                lines = f.readlines()

            current_file = None

            for line in lines:
                line = line.strip()

                if line.startswith('File Name:'):
                    current_file = line.split(': ')[1]
                    file_info[current_file] = {'seizures': [], 'num_seizures': 0}

                elif line.startswith('Number of Seizures in File:'):
                    try:
                        num_seizures = int(line.split(': ')[1])
                    except:
                        num_seizures = 0
                    if current_file:
                        file_info[current_file]['num_seizures'] = num_seizures

                elif "Seizure" in line and "Start Time" in line:
                    if current_file:
                        start_time = int(line.split(":")[1].split()[0])
                        file_info[current_file]["seizures"].append({"start": start_time})

                elif "Seizure" in line and "End Time" in line:
                    if current_file and file_info[current_file]["seizures"]:
                        end_time = int(line.split(":")[1].split()[0])
                        file_info[current_file]["seizures"][-1]["end"] = end_time

        return file_info

    # ======================================================
    # EDF LOADER
    # ======================================================
    def load_edf_file(self, filename):

        try:
            raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
            raw.filter(0.5, 40, verbose=False)
            raw.notch_filter(50, verbose=False)

            data = raw.get_data().astype(np.float32)

            if data.shape[0] > self.n_channels:
                data = data[:self.n_channels, :]
            elif data.shape[0] < self.n_channels:
                pad_width = ((0, self.n_channels - data.shape[0]), (0, 0))
                data = np.pad(data, pad_width, mode='constant')

            data = (data - np.mean(data, axis=1, keepdims=True)) / \
                   (np.std(data, axis=1, keepdims=True) + 1e-6)

            return data, raw.info['sfreq']

        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return None, None

    # ======================================================
    # WINDOW CREATION
    # ======================================================
    def create_windows(self, data, seizure_times=None):

        n_channels, n_samples = data.shape
        window_samples = int(self.window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - self.overlap))

        windows, labels, timestamps = [], [], []

        for start in range(0, n_samples - window_samples + 1, step_samples):

            end = start + window_samples
            window_data = data[:, start:end]

            window_time_start = start / self.sampling_rate
            window_time_end = end / self.sampling_rate

            label = 0

            if seizure_times:
                for seizure in seizure_times:
                    s_start = seizure.get('start', None)

                    if s_start is None:
                        continue

                    preictal_start = max(0, s_start - self.preictal_horizon)

                    if (window_time_start >= preictal_start) and (window_time_end <= s_start):
                        label = 1
                        break

            windows.append(window_data)
            labels.append(label)
            timestamps.append(window_time_start)

        return np.array(windows, dtype=np.float32), \
               np.array(labels, dtype=np.int32), \
               np.array(timestamps, dtype=np.float32)

    # ======================================================
    # DATASET PREP
    # ======================================================
    def prepare_dataset(self, target_patient):

        print("Parsing summary files...")
        file_info = self._parse_all_summaries()

        seizure_files, non_seizure_files = [], []

        for fname, info in file_info.items():

            if not fname.startswith(target_patient):
                continue

            if info.get("num_seizures", 0) > 0:
                seizure_files.append(fname)
            else:
                non_seizure_files.append(fname)

        print("Seizure EDF files:", len(seizure_files))
        print("Non-Seizure EDF files:", len(non_seizure_files))

        if len(seizure_files) < 2:
            raise ValueError("Not enough seizure EDF files for file-level split!")

        random.shuffle(seizure_files)

        test_files = [seizure_files[-1]] + non_seizure_files[-2:]
        train_files = seizure_files[:-1]

        random.shuffle(non_seizure_files)
        train_files += non_seizure_files[:len(train_files) * 2]

        X_train, y_train, pid_train = [], [], []

        print("\n--- Creating TRAIN windows ---")

        for filename in train_files:

            patient_folder = filename.split("_")[0]
            file_path = os.path.join(self.data_path, patient_folder, filename)

            if not os.path.exists(file_path):
                continue

            data, fs = self.load_edf_file(file_path)
            if data is None:
                continue

            seizure_times = file_info.get(filename, {}).get("seizures", None)

            windows, labels, _ = self.create_windows(data, seizure_times)

            if windows.size == 0:
                continue

            patient_id = int(patient_folder.replace("chb", "")) - 1

            X_train.extend(windows)
            y_train.extend(labels)
            pid_train.extend([patient_id] * len(windows))

        if len(X_train) == 0:
            raise RuntimeError("No TRAIN windows created!")

        print("Train seizure windows:", np.sum(y_train))

        return np.array(X_train), np.array(y_train), np.array(pid_train)

    # ======================================================
    # TRANSFORMER BLOCK 
    # ======================================================
    def transformer_block(self, x, num_heads=2, key_dim=16,
                          ff_dim=32, dropout=0.3):

        attn_layer = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )

        attn = attn_layer(x, x)
        attn = Dropout(dropout)(attn)

        x = Add()([x, attn])
        x = LayerNormalization()(x)

        ffn = Dense(ff_dim, activation='relu')(x)
        ffn = Dense(x.shape[-1])(ffn)
        ffn = Dropout(dropout)(ffn)

        x = Add()([x, ffn])
        x = LayerNormalization()(x)

        return x

    # ======================================================
    # CNN MODEL BUILD 
    # ======================================================
    def build_cnn_model(self, input_shape):

        eeg_input = Input(shape=input_shape)
        patient_input = Input(shape=(1,), name="patient_id")

        patient_embed = Embedding(input_dim=24, output_dim=8)(patient_input)
        patient_embed = Flatten()(patient_embed)

        b1 = SeparableConv2D(16, (3, 3), padding='same', activation='relu')(eeg_input)
        b1 = BatchNormalization()(b1)
        b1 = MaxPooling2D((4, 2))(b1)
        b1 = Dropout(0.3)(b1)

        b2 = SeparableConv2D(16, (7, 3), padding='same', activation='relu')(eeg_input)
        b2 = BatchNormalization()(b2)
        b2 = MaxPooling2D((4, 2))(b2)
        b2 = Dropout(0.3)(b2)

        b3 = SeparableConv2D(16, (11, 3), padding='same', activation='relu')(eeg_input)
        b3 = BatchNormalization()(b3)
        b3 = MaxPooling2D((4, 2))(b3)
        b3 = Dropout(0.3)(b3)

        x = Concatenate(axis=-1)([b1, b2, b3])

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((4, 2))(x)
        x = Dropout(0.4)(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.4)(x)

        # TRANSFORMER STAYS
        x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

        x = self.transformer_block(
            x,
            num_heads=2,
            key_dim=16,
            ff_dim=32
        )

        x = GlobalAveragePooling1D()(x)

        fusion = Concatenate()([x, patient_embed])

        fusion = Dense(128, activation='relu')(fusion)
        fusion = Dropout(0.5)(fusion)

        outputs = Dense(1, activation='sigmoid')(fusion)

        model = Model(inputs=[eeg_input, patient_input], outputs=outputs)

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )

        return model

    # ======================================================
    # TRAIN MODEL
    # ======================================================
    def train_model(self, X_raw, y, patient_ids):

        X = X_raw.transpose(0, 2, 1)
        X = X[..., np.newaxis]

        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pid_train, pid_test = patient_ids[train_idx], patient_ids[test_idx]

        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        self.cnn_model = self.build_cnn_model(input_shape)

        class_weight = {0: 1.0, 1: 8.0}

        history = self.cnn_model.fit(
            [X_train, pid_train],
            y_train,
            validation_data=([X_test, pid_test], y_test),
            epochs=50,
            batch_size=32,
            class_weight=class_weight,
            verbose=1
        )

        y_prob = self.cnn_model.predict([X_test, pid_test]).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        print("\nCNN Results:")
        print(classification_report(y_test, y_pred))
        print("AUC Score:", roc_auc_score(y_test, y_prob))

        return history

    # ======================================================
    # SAVE + LOAD MODEL
    # ======================================================
    def save_model(self, model_path='seizure_cnn_model.h5'):
        if self.cnn_model:
            self.cnn_model.save(model_path)
            print(f"CNN model saved: {model_path}")
        else:
            print("No model to save!")


# ======================================================
# MAIN RUN
# ======================================================
if __name__ == "__main__":

    data_path = "/kaggle/input/datasets/abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric/chb-mit-scalp-eeg-database-1.0.0"

    predictor = EEGSeizurePredictionCNN(
        data_path=data_path,
        preictal_sec=300
    )

    target_patients = ["chb01", "chb02", "chb03", "chb04", "chb05"]

    for patient in target_patients:

        print("\n===================================")
        print(f" Training Model for Patient: {patient}")
        print("===================================")

        X_all, y_all, pid_all = predictor.prepare_dataset(patient)

        history = predictor.train_model(X_all, y_all, pid_all)

        predictor.save_model(f"seizure_model_{patient}.h5")
