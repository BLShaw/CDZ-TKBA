import sys
import os

# Add project root to sys.path so 'src' module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from src.config import Config
from src.dataset import MultimodalDataset
from src.models import Autoencoder
from src.utils.training import train_autoencoder
from src.core.brain import Brain

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting System on {device} ===")
    print(f"Config MNIST Layers: {Config.MNIST_LAYERS}")
    print(f"Config FSDD Layers: {Config.FSDD_LAYERS}")

    # 2. Load Datasets
    print("\n[1/4] Initializing Datasets...")
    ds = MultimodalDataset(root_dir='data')
    
    if len(ds.fsdd_data) == 0:
        print("Warning: FSDD data is empty. Audio modality will be silent.")

    # 3. Train/Load Autoencoders
    print("\n[2/4] Preparing Autoencoders...")
    
    # --- Visual AE ---
    visual_ae_path = 'data/visual_ae.pth'
    visual_ae = Autoencoder(Config.MNIST_LAYERS).to(device)
    
    if os.path.exists(visual_ae_path):
        print("Loading pre-trained Visual AE...")
        visual_ae.load_state_dict(torch.load(visual_ae_path, map_location=device))
    else:
        print("Training Visual AE (this may take a while)...")
        # Collect a subset of MNIST for training
        loader = torch.utils.data.DataLoader(ds.mnist_data, batch_size=1000, shuffle=True)
        mnist_samples = []
        for i, (data, _) in enumerate(loader):
            mnist_samples.append(data)
            if len(mnist_samples) * 1000 >= 10000: break 
        
        if mnist_samples:
            train_data = torch.cat(mnist_samples)
            train_autoencoder(visual_ae, train_data, Config.AE_EPOCHS_MNIST, Config.AE_BATCH_SIZE, device)
            torch.save(visual_ae.state_dict(), visual_ae_path)
            print("Visual AE saved.")

    # --- Audio AE ---
    audio_ae_path = 'data/audio_ae.pth'
    audio_ae = Autoencoder(Config.FSDD_LAYERS).to(device)
    
    if os.path.exists(audio_ae_path):
        print("Loading pre-trained Audio AE...")
        audio_ae.load_state_dict(torch.load(audio_ae_path, map_location=device))
    else:
        if len(ds.fsdd_data) > 0:
            print("Training Audio AE...")
            fsdd_tensor = torch.tensor(ds.fsdd_data)
            train_autoencoder(audio_ae, fsdd_tensor, Config.AE_EPOCHS_FSDD, Config.AE_BATCH_SIZE, device)
            torch.save(audio_ae.state_dict(), audio_ae_path)
            print("Audio AE saved.")
        else:
            print("Skipping Audio AE training (no data).")

    # 4. Initialize Brain
    print("\n[3/4] Initializing Brain Architecture...")
    brain = Brain()
    visual_cortex = brain.add_cortex('visual', visual_ae)
    audio_cortex = brain.add_cortex('audio', audio_ae)

    # 5. Run Simulation
    print("\n[4/4] Running Simulation...")
    visual_ae.eval()
    audio_ae.eval()
    
    SIMULATION_STEPS = 5000 
    print(f"Simulating {SIMULATION_STEPS} timesteps. Press Ctrl+C to stop early.")
    
    try:
        for t in range(SIMULATION_STEPS):
            brain.increment_timestep()
            
            # A. Get Environment Data
            vis_tensor, aud_tensor, label = ds.get_paired_sample()
            
            # B. Encode
            # Flatten inputs for MLP
            vis_flat = vis_tensor.view(-1).to(device)
            aud_flat = aud_tensor.view(-1).float().to(device)
            
            with torch.no_grad():
                # Get embeddings from AE and convert to numpy for the Brain
                vis_encoding = visual_ae.get_encoding(vis_flat).cpu().numpy()
                
                if len(ds.fsdd_data) > 0:
                    aud_encoding = audio_ae.get_encoding(aud_flat).cpu().numpy()
                else:
                    aud_encoding = np.zeros(Config.FSDD_LAYERS[-1])

            # C. Send to Cortices
            # We pass the NUMPY arrays here. Cortex must handle them.
            brain.receive_sensory_input(visual_cortex, vis_encoding)
            if len(ds.fsdd_data) > 0:
                brain.receive_sensory_input(audio_cortex, aud_encoding)
            
            # D. Brain Cycle
            brain.cleanup()
            brain.create_new_nodes()
            
            # E. Logging
            if t % 100 == 0:
                v_nodes = len(visual_cortex.node_manager.nodes)
                a_nodes = len(audio_cortex.node_manager.nodes)
                print(f"Time: {t:4d} | Label: {label} | V-Nodes: {v_nodes:3d} | A-Nodes: {a_nodes:3d}")
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    print("\n=== Simulation Complete ===")
    brain.cleanup(force=True)
    print("Final Cleanup Done.")

if __name__ == "__main__":
    main()