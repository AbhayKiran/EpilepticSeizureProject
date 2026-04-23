from django.conf import settings
import os

# predict_seizure.py
import numpy as np
import mne
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse
import warnings
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

class EEGSeizurePredictor:

    def __init__(self, models_dir=None):

            if models_dir is None:
                models_dir = os.path.join(
                    os.path.dirname(__file__),  # predictor folder
                    "models"                    # models folder inside predictor
                )

            self.models_dir = models_dir
            self.model = None
            self.attention_model = None


            self.sampling_rate = 256
            self.window_size = 10
            self.overlap = 0.25
            self.n_channels = 23

        
    def load_model(self):

        model_path = os.path.join(self.models_dir, "seizure_cnn_colab.h5")

        if not os.path.exists(model_path):
            raise FileNotFoundError("❌ Model not found")

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'sum_over_time': lambda x: tf.reduce_sum(x, axis=1)
            },
            compile=False
        )

        # 🔥 Extract attention from same model
        self.attention_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer("multi_head_attention_3").output
        )

        print("✅ Model + Attention ready")
    
    def load_edf_file(self, filename):
        try:
            print(f"📂 Loading {filename}...")
            raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
            fs = raw.info['sfreq']

            # Fix sampling rate mismatch
            if int(fs) != 256:
                print(f"⚠ Warning: Sampling rate = {fs}, resampling to 256 Hz")
                raw.resample(256)
                fs = 256
            
            data = raw.get_data().astype(np.float32)

            # Replace NaN/inf with 0 to avoid CNN breaking
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply the same filters as training
            raw.filter(0.5, 40, verbose=False)
            raw.notch_filter(50, verbose=False)

            # Reload filtered data
            data = raw.get_data().astype(np.float32)

            # Normalize same as training (per channel z-score)
            data = (data - np.mean(data, axis=1, keepdims=True)) / \
                (np.std(data, axis=1, keepdims=True) + 1e-6)
                
            # Pad / trim to exactly 23 channels
            if data.shape[0] > self.n_channels:
                data = data[:self.n_channels, :]
            elif data.shape[0] < self.n_channels:
                pad_width = ((0, self.n_channels - data.shape[0]), (0, 0))
                data = np.pad(data, pad_width, mode='constant')

            duration = data.shape[1] / 256 / 3600
            print(f"   Duration: {duration:.2f} hours")
            print(f"   Shape: {data.shape}")

            return data, fs
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")
            return None, None
    
    def create_windows(self, data):
        """Create overlapping windows from EEG data"""
        n_channels, n_samples = data.shape
        window_samples = int(self.window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - self.overlap))
        
        windows = []
        timestamps = []
        
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window_data = data[:, start:end]
            
            window_time_start = start / self.sampling_rate
            windows.append(window_data)
            timestamps.append(window_time_start)
        
        return np.array(windows), np.array(timestamps)
    
    def get_attention(self, X, patient_id):

        attn = self.attention_model.predict([X, patient_id], verbose=0)

        if isinstance(attn, list) or isinstance(attn, tuple):
            attn = attn[1]
            
        attn = np.array(attn)

        print("ATTN SHAPE:", attn.shape)
        # expected: (batch, heads, time, time)

        # -----------------------------
        # 1. Average heads
        # -----------------------------
        attn_mean = np.mean(attn, axis=1)   # (batch, time, time)

        # -----------------------------
        # 2. Temporal importance
        # -----------------------------
        importance = np.sum(attn_mean, axis=1)   # 🔥 KEY
        # (batch, time)

        # -----------------------------
        # 3. Collapse per window
        # -----------------------------
        importance = np.mean(importance, axis=-1)   # (batch,)

        # -----------------------------
        # 4. Normalize
        # -----------------------------
        importance = (importance - np.min(importance)) / \
                    (np.max(importance) - np.min(importance) + 1e-8)

        # -----------------------------
        # 5. Smooth (VERY IMPORTANT)
        # -----------------------------
        importance = gaussian_filter1d(importance, sigma=2)

        return importance
        
    def predict_seizures(self, edf_file_path):

        # Load correct model
        if self.model is None:
            self.load_model()

        # Load EEG
        data, fs = self.load_edf_file(edf_file_path)

        windows, timestamps = self.create_windows(data)

        # CNN input shape
        X = windows.transpose(0, 2, 1)
        X = X[..., np.newaxis]

        # ✅ Correct Patient Extraction
        base = os.path.basename(edf_file_path)
        parts = base.split("_")

        patient = None
        for p in parts:
            if p.startswith("chb"):
                patient = p
                break

        if patient is None:
            raise ValueError("❌ Patient ID not found in filename!")

        pid = int(patient.replace("chb", "")) - 1
        patient_id = np.full((X.shape[0], 1), pid)

        # Predict
        probabilities = self.model.predict(
            [X, patient_id],
            verbose=0
        ).flatten()

        THRESHOLD = 0.58
        predictions = (probabilities > THRESHOLD).astype(int)
        attention = self.get_attention(X, patient_id)

        importance = attention

        return predictions, probabilities, timestamps, importance



    
    def analyze_risk(self, probabilities, timestamps, time_window=60):
        """Analyze risk in larger time bins"""
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
        
        return time_groups
    
    def plot_prediction(self, edf_file_path, predictions, probabilities, timestamps, importance):
        """Plot seizure predictions"""
        plt.style.use('default')
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Probability over time
        plt.subplot(2, 1, 1)
        hours = np.array(timestamps) / 3600
        plt.plot(hours, probabilities, 'b-', alpha=0.7, linewidth=1, label='Seizure Probability')
        plt.axhline(y=0.58, color='r', linestyle='--', label='Threshold (0.58)')
        plt.fill_between(hours, 0, probabilities, where=(probabilities > 0.58),
                         color='red', alpha=0.3, label='Predicted Seizure')
        plt.xlabel('Time (hours)')
        plt.ylabel('Probability')
        plt.title(f'CNN Seizure Prediction - {os.path.basename(edf_file_path)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Hourly risk summary
        plt.subplot(2, 1, 2)
        time_groups = self.analyze_risk(probabilities, timestamps)
        if time_groups:
            group_hours = [g['start_time']/3600 for g in time_groups]
            avg_risks = [g['avg_risk'] for g in time_groups]
            max_risks = [g['max_risk'] for g in time_groups]
            
            plt.bar(group_hours, avg_risks, width=0.8, alpha=0.7, 
                   color='skyblue', label='Avg Risk')
            plt.plot(group_hours, max_risks, 'r^-', markersize=6, linewidth=2, label='Peak Risk')
            plt.xlabel('Time (hours)')
            plt.ylabel('Risk Score')
            plt.title('Hourly Seizure Risk Summary')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_name = f"{os.path.splitext(os.path.basename(edf_file_path))[0]}_prediction_{timestamp}.png"

        plot_path = os.path.join(settings.STATICFILES_DIRS[0], plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # ---- Separate Attention Graph ----
        plt.figure(figsize=(12, 4))

        hours = np.array(timestamps) / 3600

        plt.plot(hours, importance, color='red', linewidth=1.5, label='Attention Importance')

        # Highlight high attention regions
        threshold_attn = np.percentile(importance, 90)

        plt.fill_between(hours, 0, importance,
                        where=(importance > threshold_attn),
                        color='red', alpha=0.3,
                        label='High Attention Region')

        plt.xlabel("Time (hours)")
        plt.ylabel("Importance")
        plt.title(f"Model Attention (Interpretability) - {os.path.basename(edf_file_path)}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save attention plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attn_plot_name = f"{os.path.splitext(os.path.basename(edf_file_path))[0]}_attention_{timestamp}.png"

        attn_plot_path = os.path.join(settings.STATICFILES_DIRS[0], attn_plot_name)
        plt.savefig(attn_plot_path, dpi=300, bbox_inches='tight')

        print(f"📊 Attention plot saved: {attn_plot_name}")
        plt.show()

        return {
            "static_url": f"/static/{plot_name}",
            "attention_url": f"/static/{attn_plot_name}",
            "inline_data": ""
        }


    
    def print_summary(self, edf_file_path, predictions, probabilities, timestamps):
        """Print detailed prediction summary"""
                # ============================
        #    HIGH-RISK WINDOWS
        # ============================

        high_risk_mask = probabilities > 0.58
        high_times = timestamps[high_risk_mask] / 3600  # convert to hours

        if len(high_times) > 0:

            # -------------------------------------------------------
            # 1) GROUP HIGH-RISK WINDOWS INTO CLUSTERS (SEIZURE ZONES)
            # -------------------------------------------------------
            gaps = np.diff(high_times)
            blocks = []
            current_block = [high_times[0]]

            for i, g in enumerate(gaps):
                if g < 0.01:   # <0.01 hr (~0.6 min) → same cluster
                    current_block.append(high_times[i+1])
                else:
                    blocks.append(current_block)
                    current_block = [high_times[i+1]]
            blocks.append(current_block)

            # Largest block = actual seizure zone
            seizure_block = max(blocks, key=len)
            seizure_start = seizure_block[0]
            seizure_end = seizure_block[-1]

            print(f"\n🧠 ACTUAL SEIZURE (estimated):")
            print(f"   Start: {seizure_start:.3f} hr")
            print(f"   End:   {seizure_end:.3f} hr")

            # -------------------------------------------------------
            # 2) EXTRACT ONLY REAL PRE-ALERTS 
            #    (before seizure, ignoring post-seizure & far alerts)
            # -------------------------------------------------------
            pre_alerts = []

            for t in high_times:
                diff_hr = seizure_start - t
                diff_min = diff_hr * 60

                # Keep only alerts before seizure and not too far away
                if 0 < diff_min <= 5:
                    pre_alerts.append((t, diff_min))

            print(f"\n⏳ PRE-ALERTS:")

            if len(pre_alerts) == 0:
                print("   No meaningful early warnings detected.")
            else:
                for t, m in pre_alerts:
                    print(f"   {t:.3f} hr → {m:.1f} min before seizure")

        else:
            print("\n⚠️ No high-risk periods detected.")

        print("="*60)

    
    def predict_file(self, edf_file_path, show_plot=True):
        """Main prediction function"""
        print(f"\n Starting prediction for: {edf_file_path}")
        
        predictions, probabilities, timestamps, importance = self.predict_seizures(edf_file_path)
        if predictions is None:
            print("❌ Prediction failed!")
            return
        
        # Print summary
        self.print_summary(edf_file_path, predictions, probabilities, timestamps)
        
        # Plot results
        if show_plot:
            self.plot_prediction(edf_file_path, predictions, probabilities, timestamps, importance)
        
        print(f"\n✅ Prediction completed!")
        return predictions, probabilities, timestamps, importance


# =============== COMMAND LINE USAGE ===============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print("❌ File not found")
        return

    # ✅ Personalized predictor
    predictor = EEGSeizurePredictor()

    predictor.predict_file(args.filename)



if __name__ == "__main__":
    main()
