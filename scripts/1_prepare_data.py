import sys
import os
import glob
import subprocess
import shutil
import numpy as np
import torch
import scipy.io.wavfile as wav
import scipy.signal
from torchvision import datasets, transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trim_silence(audio, threshold=0.01):
    if len(audio) == 0: return audio
    max_val = np.max(np.abs(audio))
    if max_val == 0: return audio
    norm_audio = np.abs(audio) / max_val
    mask = norm_audio > threshold
    indices = np.where(mask)[0]
    if len(indices) == 0: return audio
    start = indices[0]
    end = indices[-1]
    start = max(0, start - 100)
    end = min(len(audio), end + 100)
    return audio[start:end]

def prepare_data():
    print("=== Step 1: Preparing Datasets ===")
    
    # Paths
    root_dir = 'data'
    processed_dir = os.path.join(root_dir, 'processed')
    spectrograms_dir = os.path.join(root_dir, 'spectrograms')
    mnist_dir = os.path.join(root_dir, 'mnist')
    fsdd_dir = os.path.join(root_dir, 'fsdd')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(spectrograms_dir, exist_ok=True)
    
    # --- 1. MNIST ---
    print("\n[MNIST] Checking/Downloading...")
    # We use torchvision to handle download
    mnist_train_ds = datasets.MNIST(root=mnist_dir, train=True, download=True, transform=transforms.ToTensor())
    mnist_test_ds = datasets.MNIST(root=mnist_dir, train=False, download=True, transform=transforms.ToTensor())
    
    print("[MNIST] Processing and saving to .npy...")
    def process_mnist(ds, name):
        loader = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=False)
        data_list = []
        label_list = []
        for data, targets in loader:
            data_list.append(data.numpy())
            label_list.append(targets.numpy())
        
        X = np.concatenate(data_list)
        y = np.concatenate(label_list)
        
        np.save(os.path.join(processed_dir, f'mnist_{name}_data.npy'), X)
        np.save(os.path.join(processed_dir, f'mnist_{name}_labels.npy'), y)
        print(f"  Saved mnist_{name}: {X.shape}")

    process_mnist(mnist_train_ds, 'train')
    process_mnist(mnist_test_ds, 'test')
    
    # --- 2. FSDD ---
    print("\n[FSDD] Checking/Downloading...")
    
    # Check for existing wav files first
    wav_files = glob.glob(os.path.join(fsdd_dir, 'recordings', '*.wav'))
    if not wav_files:
        wav_files = glob.glob(os.path.join(fsdd_dir, '*.wav'))

    # If no files, attempt download
    if not wav_files:
        if os.path.exists(fsdd_dir):
            # Directory exists but no wavs. Is it a git repo?
            if os.path.exists(os.path.join(fsdd_dir, '.git')):
                print("  FSDD dir exists and is a git repo. Pulling...")
                try:
                    subprocess.check_call(['git', '-C', fsdd_dir, 'pull'])
                except Exception as e:
                     print(f"  Error pulling FSDD: {e}")
            else:
                print("  FSDD dir exists but is not a valid git repo (and has no wavs). Removing and re-cloning...")
                try:
                    shutil.rmtree(fsdd_dir)
                except Exception as e:
                    print(f"  Error removing broken FSDD dir: {e}. Please delete 'data/fsdd' manually.")
                    return

        if not os.path.exists(fsdd_dir):
            print(f"  Cloning FSDD to {fsdd_dir}...")
            try:
                subprocess.check_call(['git', 'clone', 'https://github.com/Jakobovski/free-spoken-digit-dataset.git', fsdd_dir])
            except Exception as e:
                print(f"  Error cloning FSDD: {e}")
                print("  Please manually download FSDD recordings to data/fsdd/recordings/")
                return

    # Check again after potential download
    wav_files = glob.glob(os.path.join(fsdd_dir, 'recordings', '*.wav'))
    if not wav_files:
        wav_files = glob.glob(os.path.join(fsdd_dir, '*.wav'))
        
    if not wav_files:
        print("  No wav files found even after clone attempt. Skipping FSDD.")
        return

    print(f"[FSDD] Found {len(wav_files)} files. Generating Spectrograms...")
    
    # Prepare lists
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    
    for f_path in wav_files:
        try:
            filename = os.path.basename(f_path)
            # Format: {digit}_{speaker}_{index}.wav
            parts = filename.split('_')
            if len(parts) < 3: continue
            
            digit = int(parts[0])
            speaker = parts[1]
            idx_str = parts[2].split('.')[0]
            idx = int(idx_str)
            
            # Load
            rate, samples = wav.read(f_path)
            samples = samples.astype(np.float32)
            
            # Trim
            samples = trim_silence(samples)
            
            # Spectrogram
            f_freq, t_time, Sxx = scipy.signal.spectrogram(samples, fs=rate, nperseg=256, noverlap=128)
            Sxx = np.log(Sxx + 1e-10)
            
            # Normalize [0, 1]
            s_min, s_max = Sxx.min(), Sxx.max()
            if s_max - s_min > 0:
                Sxx = (Sxx - s_min) / (s_max - s_min)
            else:
                Sxx = np.zeros_like(Sxx)
                
            # Resize to 64x64
            Sxx_t = torch.tensor(Sxx).unsqueeze(0).unsqueeze(0)
            Sxx_t = F.interpolate(Sxx_t, size=(64, 64), mode='bilinear', align_corners=False)
            spec_np = Sxx_t.squeeze().numpy() # 64x64
            
            # Save Individual Spectrogram
            # Split logic: 0-4 test, 5+ train
            is_test = idx <= 4
            subset_name = 'test' if is_test else 'train'
            
            subset_spec_dir = os.path.join(spectrograms_dir, subset_name)
            os.makedirs(subset_spec_dir, exist_ok=True)
            
            save_name = f"{digit}_{speaker}_{idx}.npy"
            np.save(os.path.join(subset_spec_dir, save_name), spec_np)

            # Save as PNG
            png_name = f"{digit}_{speaker}_{idx}.png"
            plt.imsave(os.path.join(subset_spec_dir, png_name), spec_np, cmap='gray')
            
            # Add to Aggregated List
            # We keep the (1, 64, 64) shape for the AE input (N, 1, 64, 64)
            spec_tensor_np = Sxx_t.numpy() 
            
            if is_test:
                test_data.append(spec_tensor_np)
                test_labels.append(digit)
            else:
                train_data.append(spec_tensor_np)
                train_labels.append(digit)
                
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            
    # Save Aggregated
    if train_data:
        X_train = np.concatenate(train_data, axis=0)
        y_train = np.array(train_labels)
        np.save(os.path.join(processed_dir, 'fsdd_train_data.npy'), X_train)
        np.save(os.path.join(processed_dir, 'fsdd_train_labels.npy'), y_train)
        print(f"  Saved fsdd_train: {X_train.shape}")
        
    if test_data:
        X_test = np.concatenate(test_data, axis=0)
        y_test = np.array(test_labels)
        np.save(os.path.join(processed_dir, 'fsdd_test_data.npy'), X_test)
        np.save(os.path.join(processed_dir, 'fsdd_test_labels.npy'), y_test)
        print(f"  Saved fsdd_test: {X_test.shape}")

if __name__ == "__main__":
    prepare_data()
