import sys
import os
import glob
import subprocess
import numpy as np
import torch
import scipy.io.wavfile as wav
import scipy.signal
from torchvision import datasets, transforms
from torch.nn import functional as F

class MultimodalDataset:
    def __init__(self, root_dir='data'):
        self.root_dir = root_dir
        self.mnist_dir = os.path.join(root_dir, 'mnist')
        self.fsdd_root = os.path.join(root_dir, 'fsdd') 
        
        # Only init data structures, don't load everything unless accessed
        self.mnist_data = None
        self.fsdd_data = None
        self.fsdd_labels = None
        
        print("Checking MNIST dataset...")
        self.mnist_data = datasets.MNIST(
            root=self.mnist_dir, 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        
        print("Checking FSDD dataset...")
        self.fsdd_data, self.fsdd_labels = self._load_fsdd()
        
    def _trim_silence(self, audio, threshold=0.01):
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

    def _load_fsdd(self):
        recordings_dir = os.path.join(self.fsdd_root, 'recordings')
        files = glob.glob(os.path.join(recordings_dir, '*.wav'))
        if not files:
            files = glob.glob(os.path.join(self.fsdd_root, '*.wav'))
            
        if not files:
            print(f"No .wav files found in {self.fsdd_root} or {recordings_dir}.")
            print("Attempting clone (if directory empty)...")
            if not os.path.exists(self.fsdd_root) or not os.listdir(self.fsdd_root):
                try:
                    subprocess.check_call(['git', 'clone', 'https://github.com/Jakobovski/free-spoken-digit-dataset.git', self.fsdd_root])
                    files = glob.glob(os.path.join(self.fsdd_root, 'recordings', '*.wav'))
                except Exception as e:
                    print(f"Error cloning FSDD: {e}")
                    return np.array([]), np.array([])
            else:
                print("Directory exists but contains no wav files. Skipping download.")
                return np.array([]), np.array([])
            
        data = []
        labels = []
        
        print(f"Processing {len(files)} FSDD audio files...")
        for f in files:
            try:
                filename = os.path.basename(f)
                parts = filename.split('_')
                if len(parts) >= 1 and parts[0].isdigit():
                    digit = int(parts[0])
                else:
                    continue 
                
                rate, samples = wav.read(f)
                samples = samples.astype(np.float32)
                samples = self._trim_silence(samples)
                f_freq, t_time, Sxx = scipy.signal.spectrogram(samples, fs=rate, nperseg=256, noverlap=128)
                Sxx = np.log(Sxx + 1e-10)
                
                s_min = Sxx.min()
                s_max = Sxx.max()
                if s_max - s_min > 0:
                    Sxx = (Sxx - s_min) / (s_max - s_min)
                else:
                    Sxx = np.zeros_like(Sxx)
                
                Sxx_t = torch.tensor(Sxx).unsqueeze(0).unsqueeze(0)
                Sxx_t = F.interpolate(Sxx_t, size=(64, 64), mode='bilinear', align_corners=False)
                
                data.append(Sxx_t.squeeze().numpy())
                labels.append(digit)
            except Exception as e:
                pass
            
        return np.array(data), np.array(labels)

    def get_paired_sample(self):
        # Legacy method, kept if needed but not used in new pipeline
        v_idx = np.random.randint(0, len(self.mnist_data))
        visual_tensor, v_label = self.mnist_data[v_idx]
        
        if len(self.fsdd_data) > 0:
            matching_indices = np.where(self.fsdd_labels == v_label)[0]
            if len(matching_indices) > 0:
                a_idx = np.random.choice(matching_indices)
                audio_numpy = self.fsdd_data[a_idx]
            else:
                a_idx = np.random.randint(0, len(self.fsdd_data))
                audio_numpy = self.fsdd_data[a_idx]
        else:
            audio_numpy = np.zeros((64, 64), dtype=np.float32)

        return visual_tensor, torch.tensor(audio_numpy), v_label