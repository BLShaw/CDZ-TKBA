import sys
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.models import Autoencoder
from src.core.database import db

def reconstruct_audio_from_spectrogram(spec, n_iter=32, fs=8000, n_fft=256, hop_length=128):
    """
    Reconstructs time-domain audio from a magnitude spectrogram using Griffin-Lim.
    spec: (freq, time) numpy array (linear magnitude, not log)
    """
    import librosa
    
    # 1. De-Normalize and Invert Log    
    # Scale to typical log-spec range (approx 0 to 10)
    S_scaled = spec * 10.0 
    S = np.exp(S_scaled) - 1e-10
    
    # 2. Handle resizing artifacts
    import scipy.ndimage
    # Target shape: (129, Time)
    target_shape = (129, 64)
    S_resized = scipy.ndimage.zoom(S, (target_shape[0]/S.shape[0], target_shape[1]/S.shape[1]))
    
    # 3. Griffin-Lim
    try:
        audio = librosa.griffinlim(S_resized, n_iter=n_iter, win_length=n_fft, hop_length=hop_length)
        
        # 4. Normalize Audio Volume (Maximize to -1..1)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
    except ImportError:
        print("librosa not found. Falling back to dummy audio.")
        return np.zeros(8000)
        
    return audio

def demo_generation():
    print("=== Step 6: Generative Association Demo ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for librosa
    try:
        import librosa
    except ImportError:
        print("WARNING: 'librosa' not found. Cannot save .wav files. Run 'pip install librosa'.")
    
    # 1. Load Brain
    brain_path = 'data/brain.pkl'
    if not os.path.exists(brain_path):
        print("Brain not found. Run Step 3 first.")
        return

    with open(brain_path, 'rb') as f:
        data_dump = pickle.load(f)
        if isinstance(data_dump, tuple):
            brain, db_state = data_dump
            # Restore DB
            db.nodes.data = db_state['nodes']
            db.clusters.data = db_state['clusters']
            db.nodes_to_clusters.data = db_state['nodes_to_clusters']
            db.clusters_to_nodes.data = db_state['clusters_to_nodes']
            db.node_manager_to_nodes.data = db_state['node_manager_to_nodes']
        else:
            brain = data_dump
            
    v_cortex = brain.get_cortex('visual')
    a_cortex = brain.get_cortex('audio')
    
    # 2. Load Autoencoders (for decoding)
    print("Loading Autoencoders...")
    visual_ae = Autoencoder(Config.MNIST_LAYERS).to(device)
    visual_ae.load_state_dict(torch.load('data/visual_ae.pth', map_location=device))
    visual_ae.eval()
    
    audio_ae = Autoencoder(Config.FSDD_LAYERS).to(device)
    audio_ae.load_state_dict(torch.load('data/audio_ae.pth', map_location=device))
    audio_ae.eval()
    
    # 3. Load Test Data (to pick a sample)
    enc_dir = 'data/encodings'
    X_v_test = np.load(os.path.join(enc_dir, 'visual_test_encodings.npy'))
    y_v_test = np.load(os.path.join(enc_dir, 'visual_test_labels.npy'))
    
    # Output dir
    out_dir = 'data/demo_outputs'
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Scenario 1: Image -> Sound ---
    print("\n[Scenario 1] Vision -> Audio Generation")
    # Pick random samples for each digit
    for digit in range(10):
        indices = np.where(y_v_test == digit)[0]
        if len(indices) == 0: continue
        
        idx = indices[0] # First one
        sample_enc = X_v_test[idx]
        
        # A. Visual Cortex Fires
        v_cluster = v_cortex.receive_sensory_input(sample_enc, learn=False)
        
        if not v_cluster:
            print(f"Digit {digit}: No visual cluster fired.")
            continue
            
        # B. CDZ Association
        # Get the strongest correlated cluster in the Audio Cortex
        # v_cluster is in Visual. We check CDZ correlations.
        cdz_conn = brain.cdz.correlations.get(v_cluster.name)
        
        if cdz_conn:
            a_cluster, strength = cdz_conn.get_strongest_correlation()
            # a_cluster is an Audio Cluster object
            # Get the "prototype" node for this cluster
            a_node = a_cluster.get_strongest_node()
            
            if a_node:
                # C. Decode Audio
                audio_enc = a_node.position
                audio_recon = audio_ae.decode(audio_enc).cpu().numpy().reshape(64, 64)
                
                # Save inputs and outputs
                fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                
                # Reconstruct Input Image for sanity check
                vis_recon = visual_ae.decode(sample_enc).cpu().numpy().reshape(28, 28)
                ax[0].imshow(vis_recon, cmap='gray')
                ax[0].set_title(f"Input: Visual (Digit {digit})")
                ax[0].axis('off')
                
                # Generated Audio Spec
                ax[1].imshow(audio_recon, cmap='viridis')
                ax[1].set_title(f"Generated: Audio {digit}")
                ax[1].axis('off')
                
                plt.suptitle(f"Cross-Modal Generation: Digit {digit}")
                plt.savefig(os.path.join(out_dir, f"gen_vis2aud_digit_{digit}.png"))
                plt.close()
                
                # Save .wav file
                try:
                    audio_waveform = reconstruct_audio_from_spectrogram(audio_recon)
                    wav_path = os.path.join(out_dir, f"gen_vis2aud_digit_{digit}.wav")
                    wav.write(wav_path, 8000, audio_waveform)
                    print(f"  Generated Audio Spectrogram and WAV for Digit {digit}")
                except Exception as e:
                    print(f"  Generated Spectrogram (Audio gen failed: {e})")
            else:
                print(f"  Digit {digit}: Associated Audio Cluster empty.")
        else:
            print(f"  Digit {digit}: No CDZ association found.")

    # --- Scenario 2: Sound -> Image ---
    print("\n[Scenario 2] Audio -> Vision Generation")
    
    # Load Audio Test Data if available
    if os.path.exists(os.path.join(enc_dir, 'audio_test_encodings.npy')):
        X_a_test = np.load(os.path.join(enc_dir, 'audio_test_encodings.npy'))
        y_a_test = np.load(os.path.join(enc_dir, 'audio_test_labels.npy'))
        
        for digit in range(10):
            indices = np.where(y_a_test == digit)[0]
            if len(indices) == 0: continue
            
            idx = indices[0]
            sample_enc = X_a_test[idx]
            
            # A. Audio Cortex Fires
            a_cluster = a_cortex.receive_sensory_input(sample_enc, learn=False)
            
            if not a_cluster:
                print(f"Digit {digit}: No audio cluster fired.")
                continue
                
            # B. CDZ Association (Audio -> Visual)
            cdz_conn = brain.cdz.correlations.get(a_cluster.name)
            
            if cdz_conn:
                # Get Top 3 correlations
                # cdz_conn.connections is a dict {cluster_name: strength}
                sorted_conns = sorted(cdz_conn.connections.items(), key=lambda x: x[1], reverse=True)[:3]
                
                fig, ax = plt.subplots(1, 4, figsize=(16, 4))
                
                # Input Audio
                aud_recon_input = audio_ae.decode(sample_enc).cpu().numpy().reshape(64, 64)
                ax[0].imshow(aud_recon_input, cmap='viridis', origin='lower')
                ax[0].set_title(f"Input: Audio {digit}")
                ax[0].axis('off')
                
                print(f"  Digit {digit} candidates:")
                for i, (c_name, strength) in enumerate(sorted_conns):
                    # Get cluster object
                    v_c = brain.cdz.correlations[a_cluster.name].cluster_objects[c_name]
                    v_nodes = db.get_clusters_nodes(v_c)
                    
                    if v_nodes:
                        # Medoid decoding
                        positions = np.array([n.position for n in v_nodes])
                        centroid = np.mean(positions, axis=0)
                        dists = np.linalg.norm(positions - centroid, axis=1)
                        medoid_idx = np.argmin(dists)
                        visual_enc = positions[medoid_idx]
                        
                        visual_recon = visual_ae.decode(visual_enc).cpu().numpy().reshape(28, 28)
                        
                        ax[i+1].imshow(visual_recon, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
                        ax[i+1].set_title(f"Gen {i+1}: {c_name}\n(Str: {strength:.2f})")
                        ax[i+1].axis('off')
                        print(f"    {i+1}. {c_name} (Strength: {strength:.4f}) - {len(v_nodes)} nodes")
                    else:
                        ax[i+1].axis('off')
                        ax[i+1].set_title("Empty Cluster")

                plt.suptitle(f"Audio -> Visual Generation (Top 3 Matches)")
                plt.savefig(os.path.join(out_dir, f"gen_aud2vis_digit_{digit}.png"))
                plt.close()
            else:
                print(f"  Digit {digit}: No CDZ association found.")
    else:
        print("Audio test encodings not found.")

    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    demo_generation()
