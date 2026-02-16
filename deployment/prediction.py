# predict_seizure.py
import numpy as np
import mne
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

class EEGSeizurePredictor:
    def __init__(self, model_path='seizure_cnn_model.h5'):
        self.model_path = model_path
        self.sampling_rate = 256
        self.window_size = 10  # 5 seconds
        self.overlap = 0.5    # 50% overlap
        self.n_channels = 23  # CHB-MIT standard
        
        # Load the trained model
        self.load_model()
    
    def load_model(self):
        """Load the trained CNN model"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f" Model loaded: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def load_edf_file(self, filename):
        try:
            print(f" Loading {filename}...")
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
            print(f" Error loading {filename}: {str(e)}")
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
    
    def predict_seizures(self, edf_file_path):
        """Predict seizures using CNN"""
        data, fs = self.load_edf_file(edf_file_path)
        if data is None:
            return None, None, None
        
        windows, timestamps = self.create_windows(data)
        
        # Reshape for CNN: (N, time_steps, channels)
        X = windows.transpose(0, 2, 1)  # (N, 1280, 23)
        
        print(f" Predicting on {len(windows)} windows...")
        probabilities = self.model.predict(X, verbose=0).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities, timestamps
    
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
    
    def plot_prediction(self, edf_file_path, predictions, probabilities, timestamps):
        """Plot seizure predictions"""
        plt.style.use('default')
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Probability over time
        plt.subplot(2, 1, 1)
        hours = np.array(timestamps) / 3600
        plt.plot(hours, probabilities, 'b-', alpha=0.7, linewidth=1, label='Seizure Probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        plt.fill_between(hours, 0, probabilities, where=(probabilities > 0.5),
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
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_name = f"{os.path.splitext(os.path.basename(edf_file_path))[0]}_prediction_{timestamp}.png"
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        print(f" Plot saved: {plot_name}")
        plt.show()
    
    def print_summary(self, edf_file_path, predictions, probabilities, timestamps):
        """Print detailed prediction summary"""
        total_hours = len(timestamps) * self.window_size / 3600
        seizure_windows = np.sum(predictions)
        high_risk_windows = np.sum(probabilities > 0.7)
        
        print("\n" + "="*60)
        print(f" SEIZURE PREDICTION SUMMARY")
        print("="*60)
        print(f" File: {os.path.basename(edf_file_path)}")
        print(f"  Duration: {total_hours:.2f} hours")
        print(f" Total windows: {len(probabilities):,}")
        print(f" Predicted seizure windows: {seizure_windows:,} ({seizure_windows/len(probabilities)*100:.2f}%)")
        print(f"  High-risk windows (>70%): {high_risk_windows:,} ({high_risk_windows/len(probabilities)*100:.2f}%)")
        print(f" Average probability: {np.mean(probabilities):.3f}")
        print(f" Maximum probability: {np.max(probabilities):.3f}")
        print(f" Minimum probability: {np.min(probabilities):.3f}")
        
        # High risk periods
        high_risk_mask = probabilities > 0.7
        if np.any(high_risk_mask):
            high_times = timestamps[high_risk_mask] / 3600
            print(f"\n  HIGH RISK PERIODS (hours):")
            for i, t in enumerate(high_times[:10]):  # Show first 10
                print(f"   {t:.3f}")
            if len(high_times) > 10:
                print(f"   ... and {len(high_times)-10} more")
        
        print("="*60)
    
    def predict_file(self, edf_file_path, show_plot=True):
        """Main prediction function"""
        print(f"\n Starting prediction for: {edf_file_path}")
        
        predictions, probabilities, timestamps = self.predict_seizures(edf_file_path)
        if predictions is None:
            print(" Prediction failed!")
            return
        
        # Print summary
        self.print_summary(edf_file_path, predictions, probabilities, timestamps)
        
        # Plot results
        if show_plot:
            self.plot_prediction(edf_file_path, predictions, probabilities, timestamps)
        
        print(f"\n Prediction completed!")
        return predictions, probabilities, timestamps


# =============== COMMAND LINE USAGE ===============
def main():
    parser = argparse.ArgumentParser(description='Predict seizures from EEG EDF file')
    parser.add_argument('filename', type=str, help='Path to EDF file')
    parser.add_argument('--model', type=str, default='seizure_cnn_model.h5', 
                       help='Path to trained model (default: seizure_cnn_model.h5)')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Skip plotting (faster execution)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.filename):
        print(f"❌ File not found: {args.filename}")
        return
    
    # Initialize predictor
    predictor = EEGSeizurePredictor(model_path=args.model)
    
    # Predict
    show_plot = not args.no_plot
    predictor.predict_file(args.filename, show_plot=show_plot)


if __name__ == "__main__":
    main()
