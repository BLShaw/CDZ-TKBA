import sys
import os
import pickle
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.core.database import db

def refine_associations():
    print("=== Step 7: Refinement (Freeze & Associate) ===")
    
    # 1. Load Brain
    brain_path = 'data/brain.pkl'
    if not os.path.exists(brain_path):
        print("Brain not found. Run Step 3 first.")
        return

    print("Loading Brain...")
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
    
    # 2. RESET Correlations
    print("Resetting CDZ Correlations...")
    brain.cdz.correlations = {}
    brain.cdz.packet_queue.clear()
    brain.cdz.cluster_frequencies.clear() 
    
    # LOWER Learning Rate for Refinement
    # This ensures we learn stable statistical correlations, not random noise
    brain.cdz.LEARNING_RATE = 0.001 
    
    # 3. Load Data
    enc_dir = 'data/encodings'
    v_enc = np.load(os.path.join(enc_dir, 'visual_train_encodings.npy'))
    v_labels = np.load(os.path.join(enc_dir, 'visual_train_labels.npy'))
    
    if not os.path.exists(os.path.join(enc_dir, 'audio_train_encodings.npy')):
        print("Audio encodings missing.")
        return
        
    a_enc = np.load(os.path.join(enc_dir, 'audio_train_encodings.npy'))
    a_labels = np.load(os.path.join(enc_dir, 'audio_train_labels.npy'))
    
    # 4. Run Refinement Loop
    REFINE_STEPS = 100000
    print(f"Running Refinement for {REFINE_STEPS} steps (Nodes Frozen, LR=0.001)...")
    
    for t in range(REFINE_STEPS):
        brain.increment_timestep()
        
        # Get Random Pair (Correctly Matched)
        idx_v = np.random.randint(0, len(v_enc))
        vis_input = v_enc[idx_v]
        label = v_labels[idx_v]
        
        matching = np.where(a_labels == label)[0]
        if len(matching) > 0:
            idx_a = np.random.choice(matching)
            aud_input = a_enc[idx_a]
        else:
            continue # Skip if no matching audio
        
        v_cluster = v_cortex.receive_sensory_input(vis_input, learn=False)
        a_cluster = a_cortex.receive_sensory_input(aud_input, learn=False)
        
        if v_cluster and a_cluster:
            brain.cdz.update_competitive(v_cluster, a_cluster, penalty=1.0)
            
        if t % 1000 == 0:
            print(f"Refining Step {t}/{REFINE_STEPS}")

    # 5. Save Refined Brain
    print("Saving Refined Brain...")
    
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
    print("Refined Brain saved to data/brain.pkl")

if __name__ == "__main__":
    refine_associations()
