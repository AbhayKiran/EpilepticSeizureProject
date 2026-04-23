# 01_cnn_only.py  (MODIFIED FOR MULTI-PATIENT 5-MIN PREDICTION)
import os
import random
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.layers import Softmax, Multiply, Lambda
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
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import glob

def sum_over_time(x):
        return tf.reduce_sum(x, axis=1)
    
class EEGSeizurePredictionCNN:
    def __init__(self, data_path, summaries_dir=None, preictal_sec=300, max_non_seizure_files=10, random_seed=42):
        """
        data_path: root dataset folder containing patient subfolders (e.g. dataset/chb01/...)
        summaries_dir: directory containing chbXX-summary.txt files (if None, will look in data_path)
        preictal_sec: pre-ictal horizon in seconds (5 minutes = 300)
        max_non_seizure_files: how many non-seizure EDF files to include (to limit size)
        """
        self.data_path = data_path
        self.summaries_dir = summaries_dir if summaries_dir else data_path
        self.preictal_horizon = preictal_sec
        self.max_non_seizure_files = max_non_seizure_files
        self.random_seed = random_seed

        self.cnn_model = None
        self.attn_scores = None 
        self.overlap = 0.25   # 25% overlap between windows
        self.sampling_rate = 256
        self.window_size = 10  
        self.n_channels = 23  # CHB-MIT typical
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)



    def _parse_all_summaries(self):
        """
        Reads all chb*-summary.txt files in summaries_dir and returns a dict:
        { 'chb01_03.edf': {'num_seizures': n, 'seizures': [{'start': s, 'end': e}, ...]}, ... }
        """
        file_info = {}
        summary_paths = glob.glob(os.path.join(self.summaries_dir, "*-summary.txt"))
        if not summary_paths:
            # Also accept pattern chb??-summary.txt or chb*-summary.txt
            summary_paths = glob.glob(os.path.join(self.summaries_dir, "chb*-summary.txt"))

        if not summary_paths:
            raise FileNotFoundError("No summary files found in summaries_dir. Place chbXX-summary.txt files there.")

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
                elif line.startswith('Seizure Start Time:'):
                    if current_file:
                        try:
                            start_time = int(line.split(': ')[1].split()[0])
                        except:
                            start_time = int(float(line.split(': ')[1].split()[0]))
                        file_info[current_file]['seizures'].append({'start': start_time})
                elif line.startswith('Seizure End Time:'):
                    if current_file and file_info[current_file]['seizures']:
                        try:
                            end_time = int(line.split(': ')[1].split()[0])
                        except:
                            end_time = int(float(line.split(': ')[1].split()[0]))
                        file_info[current_file]['seizures'][-1]['end'] = end_time
        return file_info

    def load_edf_file(self, filename):
        """Load EEG data from EDF file (returns data ndarray float32 and sampling rate)"""
        try:
            raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
            raw.filter(0.5, 40, verbose=False)
            raw.notch_filter(50, verbose=False) 
            
            data = raw.get_data().astype(np.float32)
            # Ensure channels count
            if data.shape[0] > self.n_channels:
                data = data[:self.n_channels, :]
            elif data.shape[0] < self.n_channels:
                pad_width = ((0, self.n_channels - data.shape[0]), (0, 0))
                data = np.pad(data, pad_width, mode='constant')
                
                # Normalize each channel
            data = (data - np.mean(data, axis=1, keepdims=True)) / \
                (np.std(data, axis=1, keepdims=True) + 1e-6)

            return data, raw.info['sfreq']
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return None, None

    def create_windows(self, data, seizure_times=None):
        """Create overlapping windows and label them as pre-ictal/ictal(1) or inter-ictal(0) using preictal horizon."""
        n_channels, n_samples = data.shape
        window_samples = int(self.window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - self.overlap))

        windows = []
        labels = []
        timestamps = []

        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window_data = data[:, start:end]
            window_time_start = start / self.sampling_rate
            window_time_end = end / self.sampling_rate
            label = 0

            if seizure_times:
                for seizure in seizure_times:
                    s_start, s_end = seizure.get('start', None), seizure.get('end', None)
                    if s_start is None:
                        continue

                    # pre-ictal start (bounded by 0)
                    preictal_start = max(0, s_start - self.preictal_horizon)

                    # If window overlaps preictal interval OR ictal interval, label=1
                    # preictal interval = [preictal_start, s_start)
                    # ictal interval = [s_start, s_end]
                    # condition: window_end >= preictal_start and window_start <= s_end AND window_end > preictal_start
                    if (window_time_end > preictal_start) and (window_time_start < s_start):
                        label = 1
                        break

            windows.append(window_data)
            labels.append(label)
            timestamps.append(window_time_start)

        return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int32), np.array(timestamps, dtype=np.float32)

    def prepare_dataset(self):
        """Prepare dataset from all parsed summary files and EDFs under patient folders"""
        print("Parsing summary files...")
        patient_ids = []
        file_info = self._parse_all_summaries()

        # classify files into seizure-containing and non-seizure
        seizure_files = []
        non_seizure_files = []
        for fname, info in file_info.items():
            if info.get('num_seizures', 0) > 0:
                seizure_files.append(fname)
            else:
                non_seizure_files.append(fname)

        # Select all seizure files
        selected_files = seizure_files.copy()

        # Add up to max_non_seizure_files non-seizure files
        if len(non_seizure_files) > 0:
            random.shuffle(non_seizure_files)
            selected_files += non_seizure_files[:self.max_non_seizure_files]

        print("Files used for training (total {}):".format(len(selected_files)))
        for f in selected_files:
            print("  ->", f)

        X_raw = []
        y = []
        file_labels = []
        timestamps_all = []

        for filename in selected_files:
            # filename like chb01_03.edf
            patient_folder = filename.split('_')[0]  # chb01
            file_path = os.path.join(self.data_path, patient_folder, filename)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path} (skipping)")
                continue

            print(f"Processing {filename} ...")
            data, fs = self.load_edf_file(file_path)
            if data is None:
                continue

            # get the seizure info for this file
            seizure_times = file_info.get(filename, {}).get('seizures', None)
            windows, labels, timestamps = self.create_windows(data, seizure_times)

            # append
            if windows.size == 0:
                continue
            X_raw.extend(windows)            # windows shape: (n_windows, channels, time)
            y.extend(labels)
            patient_id = int(filename.split('_')[0].replace('chb', '')) - 1
            patient_ids.extend([patient_id] * len(windows))
            file_labels.extend([filename] * len(windows))
            timestamps_all.extend(timestamps)

        if len(X_raw) == 0:
            raise RuntimeError("No windows were created. Check dataset paths and summary files.")

        X_raw = np.array(X_raw, dtype=np.float32)  # shape (N, channels, time) where windows are channels x samples
        # Convert to original convention used by training: (N, time_steps, channels)
        # But note: earlier code had windows with shape (channels, window_samples). We will transpose later before training.
        y = np.array(y, dtype=np.int32)
        print(f"Dataset prepared: {len(X_raw)} windows")
        print(f"Seizure (preictal/ictal) windows: {np.sum(y)} ({np.sum(y)/len(y)*100:.2f}%)")
        print(f"Non-seizure windows: {len(y) - np.sum(y)}")


        patient_ids = np.array(patient_ids, dtype=np.int32)
        return X_raw, y, patient_ids, file_labels, timestamps_all


    def transformer_block(self, x, num_heads=2, key_dim=16, ff_dim=32, dropout=0.3, return_attention=False):
        attn_layer = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )
        
        attn, attn_scores = attn_layer(
            x, x,
            return_attention_scores=True
        )

        attn = Dropout(dropout)(attn)
        x = Add()([x, attn])
        x = LayerNormalization()(x)
    
        ffn = Dense(ff_dim, activation='relu')(x)
        ffn = Dense(x.shape[-1])(ffn)
        ffn = Dropout(dropout)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
    
        if return_attention:
            return x, attn_scores
        return x


    def build_cnn_model(self, input_shape):
        """
        Multiscale 2D CNN for EEG
        input_shape = (time_steps, channels, 1)
        """
    
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

    
        # -------- Merge multiscale features --------
        x = Concatenate(axis=-1)([b1, b2, b3])

        # -------- Deeper feature extraction --------
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((4, 2))(x)
        x = Dropout(0.4)(x)
    
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.4)(x)
    
        # -------- TRANSFORMER (SAFE PLACE) --------
        # CNN output: (batch, T, C, F)
        x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

        self.time_steps_after_cnn = x.shape[1]
        self.time_resolution = self.window_size / self.time_steps_after_cnn
    
        # EEG transformer output
        x, attn_scores = self.transformer_block(
            x,
            num_heads=2,
            key_dim=16,
            ff_dim=32,
            return_attention=True
        )

        self.attn_scores = attn_scores
        
        attention_weights = Dense(1)(x)
        attention_weights = Softmax(axis=1)(attention_weights)
        
        x = Multiply()([x, attention_weights])
        x = Lambda(sum_over_time)(x)
        
        fusion = Concatenate()([x, patient_embed])

    
        # -------- CLASSIFICATION HEAD --------
        fusion = Dense(128, activation='relu')(fusion)
        fusion = Dropout(0.5)(fusion)
        fusion = Dense(64, activation='relu')(fusion)
        outputs = Dense(1, activation='sigmoid')(fusion)
        
        model = Model(
            inputs=[eeg_input, patient_input],
            outputs=outputs
        )


    
        def focal_loss(alpha=0.25, gamma=2.0):
            def loss(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
                p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                loss_val = alpha * tf.pow(1 - p_t, gamma) * bce
                return loss_val
            return loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4),
            loss=focal_loss(),
            metrics=[tf.keras.metrics.AUC(name="auc"),
                     tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        )

        self.attention_model = Model(
            inputs=[eeg_input, patient_input],
            outputs=attn_scores
        )
    
        return model

    def build_attention_model(self):
        """
        Returns a model that outputs attention scores
        """
        return self.attention_model


    def visualize_attention(self, edf_file_path):
        data, _ = self.load_edf_file(edf_file_path)
        windows, _, timestamps = self.create_windows(data)
    
        if windows.size == 0:
            print("No windows found")
            return
    
        X = windows.transpose(0, 2, 1)[..., np.newaxis]
        patient_str = os.path.basename(edf_file_path).split('_')[0]
        pid = int(patient_str.replace('chb','')) - 1
        patient_id = np.full((X.shape[0], 1), pid)
    
        attention_model = self.build_attention_model()
        attn = attention_model.predict([X, patient_id], verbose=0)
        
        # Average over heads
        attn_mean = np.mean(attn, axis=1)   # average over heads

        # Better importance extraction
        importance = np.mean(attn_mean, axis=-1)   # how much each time attends
        importance = np.mean(importance, axis=1)   # (windows,)
        
        # Normalize (VERY IMPORTANT for clean graph)
        importance = (importance - np.min(importance)) / (np.max(importance) - np.min(importance) + 1e-8)    

        # Smooth curve (moving average)
        window = 10
        importance = np.convolve(importance, np.ones(window)/window, mode='same')


        # Plot
        plt.figure(figsize=(12,4))
        time_axis = timestamps / 60  # base window time

        plt.plot(time_axis, importance, color='red')
        plt.fill_between(time_axis, importance,
                 where=(importance > 0.7 * np.max(importance)),
                 color='red', alpha=0.3,
                 label='High Attention Region')
        plt.xlabel("Time (minutes)")
        plt.ylabel("Attention Importance")
        plt.title("Temporal Attention-Based EEG Importance")
        plt.grid(True)
        plt.show()


    def train_model(self, X_raw, y):

        """Pure window-level random split"""
    
        # Convert shape
        X = X_raw.transpose(0, 2, 1)
        X = X[..., np.newaxis]
    
        print(f"Dataset shape: {X.shape}")
    
        # -------------------------------
        # WINDOW LEVEL RANDOM SPLIT
        # -------------------------------
        from sklearn.model_selection import train_test_split
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print("Train windows:", len(X_train))
        print("Test windows :", len(X_test))
    
        # -------------------------------
        # BUILD MODEL
        # -------------------------------
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        self.cnn_model = self.build_cnn_model(input_shape)
    
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4, min_lr=1e-7, mode='max')
        ]
    
        class_weight = {0: 1.0, 1: 8.0}
    
        history = self.cnn_model.fit(
            [X_train, np.zeros((len(X_train),1))],  # dummy patient id
            y_train,
            validation_data=([X_test, np.zeros((len(X_test),1))], y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
    
        # -------------------------------
        # EVALUATION
        # -------------------------------
        y_pred_prob = self.cnn_model.predict(
            [X_test, np.zeros((len(X_test),1))],
            verbose=0
        )
    
        y_pred_prob = y_pred_prob.flatten()
    
        # Tune threshold
        from sklearn.metrics import precision_recall_curve, f1_score

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        
        best_f1 = 0
        best_threshold = 0.5
        
        for t in thresholds:
            y_temp = (y_pred_prob > t).astype(int)
            f1 = f1_score(y_test, y_temp)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        print("Best Threshold (F1):", best_threshold)
    
        y_pred = (y_pred_prob > best_threshold).astype(int)
    
        print("\nWindow-Level Results:")
        print(classification_report(y_test, y_pred))
        print(f"AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")


        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        
        # Metrics
        sensitivity = TP / (TP + FN + 1e-6)
        specificity = TN / (TN + FP + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-6)
        
        print("\n=== Detailed Metrics ===")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity        : {specificity:.4f}")
        print(f"Precision          : {precision:.4f}")
        print(f"F1 Score           : {f1:.4f}")

        # ---- False Alarm Rate per Hour ----
        # Each window = 10 sec → convert to hours
        window_duration_hr = self.window_size / 3600
        
        false_alarms = FP
        total_hours = len(y_test) * window_duration_hr
        
        false_alarm_rate = false_alarms / (total_hours + 1e-6)
        
        print(f"False Alarm Rate (/hour): {false_alarm_rate:.4f}")
    
        return history

    def predict_seizures(self, edf_file_path):
        """Predict seizures using the trained model on a single EDF file"""
        data, fs = self.load_edf_file(edf_file_path)
        if data is None:
            return None, None, None

        windows, _, timestamps = self.create_windows(data, seizure_times=None)  # no labels needed
        if windows.size == 0:
            return None, None, None
        X = windows.transpose(0, 2, 1)
        X = X[..., np.newaxis]

        patient_str = os.path.basename(edf_file_path).split('_')[0]
        pid = int(patient_str.replace("chb", "")) - 1
        patient_id = np.full((X.shape[0], 1), pid)
    
        probabilities = self.cnn_model.predict(
            [X, patient_id],
            verbose=0
        )
        probabilities = probabilities.flatten()

        predictions = (probabilities > 0.3).astype(int)
        return predictions, probabilities, timestamps

    def analyze_seizure_risk(self, edf_file_path, time_window=60):
        predictions, probabilities, timestamps = self.predict_seizures(edf_file_path)
        if predictions is None:
            return None, None
        time_groups = []
        step = time_window // self.window_size
        for i in range(0, len(timestamps), step):
            end_idx = min(i + step, len(timestamps))
            group_probs = probabilities[i:end_idx]
            time_groups.append({
                'start_time': timestamps[i],
                'end_time': timestamps[end_idx - 1] + self.window_size,
                'avg_risk': np.mean(group_probs),
                'max_risk': np.max(group_probs),
                'high_risk_count': np.sum(group_probs > 0.7)
            })
        return time_groups, [g['avg_risk'] for g in time_groups]

    def plot_seizure_prediction(self, edf_file_path, save_plot=True):
        predictions, probabilities, timestamps = self.predict_seizures(edf_file_path)
        if predictions is None:
            print("Could not analyze the file")
            return

        plt.figure(figsize=(15, 8))

        # ---- Plot 1 ----
        plt.subplot(2, 1, 1)
        hours = np.array(timestamps) / 3600
        plt.plot(hours, probabilities, 'b-', alpha=0.7, label='Seizure Probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.3)')
        plt.fill_between(hours, 0, probabilities, where=(probabilities > 0.3),
                        color='red', alpha=0.3, label='Predicted Seizure')
        plt.xlabel('Time (hours)')
        plt.ylabel('Probability')
        plt.title('CNN Seizure Prediction Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ---- Plot 2 ----
        plt.subplot(2, 1, 2)
        time_groups, risk_scores = self.analyze_seizure_risk(edf_file_path)
        if time_groups:
            group_hours = [g['start_time'] / 3600 for g in time_groups]
            avg_risks = [g['avg_risk'] for g in time_groups]
            max_risks = [g['max_risk'] for g in time_groups]

            plt.bar(group_hours, avg_risks, width=0.8, alpha=0.7, label='Avg Risk')
            plt.plot(group_hours, max_risks, 'r^-', label='Peak Risk')
            plt.xlabel('Time (hours)')
            plt.ylabel('Risk Score')
            plt.title('Hourly Seizure Risk (CNN)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_plot:
            fname = os.path.basename(edf_file_path).replace('.edf', '_cnn_prediction.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {fname}")
        plt.show()

        # ---- CRITICAL RISK SUMMARY ----
        total_hours = len(timestamps) * self.window_size / 3600
        high_risk_count = np.sum(probabilities > 0.7)
        avg_prob = np.mean(probabilities)
        max_prob = np.max(probabilities)

        print("\n=== CNN Seizure Risk Summary ===")
        print(f"Recording length: {total_hours:.2f} hours")
        print(f"High-risk windows (>70%): {high_risk_count}")
        print(f"Avg probability: {avg_prob:.3f}")
        print(f"Max probability: {max_prob:.3f}")

        if max_prob > 0.70:
            print("\n⚠️⚠️ CRITICAL PRE-ICTAL DETECTION ⚠️⚠️")
            print("High seizure risk predicted in this recording!")
        else:
            print("\nNo dangerous pre-ictal activity detected.")


    def save_model(self, model_path='seizure_cnn_model.h5'):
        if self.cnn_model:
            self.cnn_model.save(model_path)
            print(f"CNN model saved: {model_path}")

            attention_path = model_path.replace(".h5", "_attention.h5")
            self.attention_model.save(attention_path)
            print(f"✅ Attention model saved: {attention_path}")
        else:
            print("No model to save!")

    def load_model(self, model_path='seizure_cnn_model.h5'):
        self.cnn_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'sum_over_time': sum_over_time
            },
            compile=False
        )
        print(f"CNN model loaded: {model_path}")


# =============== USAGE EXAMPLE ===============
if __name__ == "__main__":

    data_path = "/kaggle/input/datasets/abhaykiran/maindataset/dataset"
    summaries_dir = data_path

    predictor = EEGSeizurePredictionCNN(
        data_path=data_path,
        summaries_dir=summaries_dir,
        preictal_sec=300,
        max_non_seizure_files=8
    )


    model_path = "seizure_cnn_colab.h5"

    # 👉 CHECK if model already exists
    if os.path.exists(model_path):
        print("Loading existing model...")
        predictor.build_cnn_model((2560, 23, 1))
        predictor.load_model(model_path)
        

    else:
        print("Training model...")
        X_raw, y, patient_ids, file_labels, timestamps = predictor.prepare_dataset()
        predictor.train_model(X_raw, y)
        predictor.save_model(model_path)

    print("\nShowing attention visualization...")
    predictor.visualize_attention("/kaggle/input/datasets/abhaykiran/maindataset/dataset/chb03/chb03_01.edf")
