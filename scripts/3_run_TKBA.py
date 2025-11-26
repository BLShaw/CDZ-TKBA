import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pickle
import numpy as np
from src.core.brain import Brain
from src.config import Config
from src.models import Autoencoder

def run_hebbian():
    print("=== Step 3: Running Hebbian Learning ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Encodings
    enc_dir = 'data/encodings'
    v_enc = np.load(os.path.join(enc_dir, 'visual_train_encodings.npy'))
    v_labels = np.load(os.path.join(enc_dir, 'visual_train_labels.npy'))
    
    has_audio = os.path.exists(os.path.join(enc_dir, 'audio_train_encodings.npy'))
    if has_audio:
        a_enc = np.load(os.path.join(enc_dir, 'audio_train_encodings.npy'))
        a_labels = np.load(os.path.join(enc_dir, 'audio_train_labels.npy'))
    
    # Initialize Brain
    # We pass "None" for AEs because we are feeding pre-computed encodings directly.
    brain = Brain()
    v_cortex = brain.add_cortex('visual', None)
    a_cortex = brain.add_cortex('audio', None)
    
    # Full Run: 55,000 samples * 2 Epochs = 110,000 steps
    SIM_STEPS = 110000 
    print(f"Running for {SIM_STEPS} steps...")
    
    for t in range(SIM_STEPS):
        brain.increment_timestep()
        
        # Get Random Pair
        # Visual
        idx_v = np.random.randint(0, len(v_enc))
        vis_input = v_enc[idx_v]
        label = v_labels[idx_v]
        
        # Audio (Grounded)
        aud_input = None
        if has_audio:
            # Find matching label
            matching = np.where(a_labels == label)[0]
            if len(matching) > 0:
                idx_a = np.random.choice(matching)
                aud_input = a_enc[idx_a]
            else:
                idx_a = np.random.randint(0, len(a_enc))
                aud_input = a_enc[idx_a]
        
        # Feed Brain
        brain.receive_sensory_input(v_cortex, vis_input)
        if aud_input is not None:
            brain.receive_sensory_input(a_cortex, aud_input)
            
        brain.cleanup()
        brain.create_new_nodes()
        
        if t % 500 == 0:
            print(f"Step {t}: V-Nodes={len(v_cortex.node_manager.nodes)}, A-Nodes={len(a_cortex.node_manager.nodes)}")

    # Final cleanup
    brain.cleanup(force=True)
    
    # Save Brain AND Database
    print("Saving Brain state...")
    from src.core.database import db
    
    # We need to save the internal data of the DB tables
    db_state = {
        'nodes': db.nodes.data,
        'clusters': db.clusters.data,
        'nodes_to_clusters': db.nodes_to_clusters.data,
        'clusters_to_nodes': db.clusters_to_nodes.data,
        'node_manager_to_nodes': db.node_manager_to_nodes.data
    }
    
    sys.setrecursionlimit(10000)
    with open('data/brain.pkl', 'wb') as f:
        pickle.dump((brain, db_state), f)
    print("Brain and Database saved to data/brain.pkl")

if __name__ == "__main__":
    run_hebbian()
