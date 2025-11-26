import sys
import os
import glob
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.models import Autoencoder
from src.utils.training import train_autoencoder

def load_spectrograms_from_dir(dir_path):
    """Loads spectrogram npy files from a directory."""
    files = glob.glob(os.path.join(dir_path, '*.npy'))
    data = []
    labels = []
    
    print(f"  Loading {len(files)} spectrograms from {dir_path}...")
    for f in files:
        try:
            # Format: {digit}_{speaker}_{index}.npy
            filename = os.path.basename(f)
            parts = filename.split('_')
            if len(parts) < 3: continue
            digit = int(parts[0])
            
            spec = np.load(f) # shape (64, 64)
            # Add channel dim for consistency (1, 64, 64)
            spec = spec[np.newaxis, :, :] 
            
            data.append(spec)
            labels.append(digit)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not data:
        return np.array([]), np.array([])
        
    return np.stack(data), np.array(labels)

def generate_encodings():
    print("=== Step 2: Generating Encodings ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = 'data/processed'
    spectrograms_dir = 'data/spectrograms'
    enc_dir = 'data/encodings'
    os.makedirs(enc_dir, exist_ok=True)

    # --- MNIST ---
    print("\n[Visual] Processing MNIST...")
    X_train = np.load(os.path.join(processed_dir, 'mnist_train_data.npy'))
    X_test = np.load(os.path.join(processed_dir, 'mnist_test_data.npy'))
    y_train = np.load(os.path.join(processed_dir, 'mnist_train_labels.npy'))
    y_test = np.load(os.path.join(processed_dir, 'mnist_test_labels.npy'))

    visual_ae = Autoencoder(Config.MNIST_LAYERS).to(device)
    
    # Train
    # Flatten for MLP AE
    X_train_flat = torch.tensor(X_train).view(X_train.shape[0], -1)
    X_test_flat = torch.tensor(X_test).view(X_test.shape[0], -1)
    
    # Train on FULL dataset
    print("  Training Visual AE on full dataset...")
    train_autoencoder(visual_ae, X_train_flat, Config.AE_EPOCHS_MNIST, Config.AE_BATCH_SIZE, device)
    
    # Helper for encoding generation
    def get_encs(model, data_tensor):
        encs = []
        batch_size = 500
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size].to(device)
            e = model.get_encoding(batch).cpu().numpy()
            encs.append(e)
        return np.concatenate(encs)

    print("  Generating Visual Encodings...")
    with torch.no_grad():
        vis_enc_train = get_encs(visual_ae, X_train_flat)
        vis_enc_test = get_encs(visual_ae, X_test_flat)

    np.save(os.path.join(enc_dir, 'visual_train_encodings.npy'), vis_enc_train)
    np.save(os.path.join(enc_dir, 'visual_test_encodings.npy'), vis_enc_test)
    np.save(os.path.join(enc_dir, 'visual_train_labels.npy'), y_train)
    np.save(os.path.join(enc_dir, 'visual_test_labels.npy'), y_test)

    # --- FSDD ---
    print("\n[Audio] Processing FSDD...")
    
    # Load from spectrograms folder
    train_spec_dir = os.path.join(spectrograms_dir, 'train')
    test_spec_dir = os.path.join(spectrograms_dir, 'test')
    
    if os.path.exists(train_spec_dir):
        X_a_train, y_a_train = load_spectrograms_from_dir(train_spec_dir)
        X_a_test, y_a_test = load_spectrograms_from_dir(test_spec_dir)
        
        if len(X_a_train) > 0:
            audio_ae = Autoencoder(Config.FSDD_LAYERS).to(device)
            
            # Flatten (N, 1, 64, 64) -> (N, 4096)
            X_a_train_flat = torch.tensor(X_a_train).float().view(X_a_train.shape[0], -1)
            X_a_test_flat = torch.tensor(X_a_test).float().view(X_a_test.shape[0], -1)
            
            print("  Training Audio AE...")
            train_autoencoder(audio_ae, X_a_train_flat, Config.AE_EPOCHS_FSDD, Config.AE_BATCH_SIZE, device)
            
            print("  Generating Audio Encodings...")
            with torch.no_grad():
                aud_enc_train = get_encs(audio_ae, X_a_train_flat)
                aud_enc_test = get_encs(audio_ae, X_a_test_flat)
                
            np.save(os.path.join(enc_dir, 'audio_train_encodings.npy'), aud_enc_train)
            np.save(os.path.join(enc_dir, 'audio_test_encodings.npy'), aud_enc_test)
            np.save(os.path.join(enc_dir, 'audio_train_labels.npy'), y_a_train)
            np.save(os.path.join(enc_dir, 'audio_test_labels.npy'), y_a_test)
        else:
            print("  No FSDD spectrograms found in train dir.")
    else:
        print(f"  Spectrogram directory not found: {train_spec_dir}")
        print("  Please run Step 1 first.")

if __name__ == "__main__":
    generate_encodings()
