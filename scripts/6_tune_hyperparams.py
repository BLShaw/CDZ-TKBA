import sys
import os
import itertools
import copy
import numpy as np
import torch
import pickle
from collections import Counter

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.brain import Brain
from src.config import Config
from src.core.database import db

# 1. Define Hyperparameter Grid
param_grid = {
    'TKBA_VISUAL_SIGMA': [0.15, 0.3, 0.5],
    'TKBA_VISUAL_VIGILANCE': [0.45, 0.6, 0.75],
    'TKBA_AUDIO_SIGMA': [0.15],      # Audio is stable
    'TKBA_AUDIO_VIGILANCE': [0.45]   # Audio is stable
}

# Reduced simulation steps for tuning speed (enough to see convergence trend)
TUNING_STEPS = 20000 

def run_tuning_session(params):
    print(f"\n--- Testing Params: {params} ---")
    
    # A. Apply Params to Config (Runtime Patching)
    # We patch the Config class attributes directly
    for k, v in params.items():
        setattr(Config, k, v)
        
    # B. Reset Database (Critical!)
    # The global 'db' object retains state. We must clear it.
    db.nodes.data.clear()
    db.clusters.data.clear()
    db.nodes_to_clusters.data.clear()
    db.clusters_to_nodes.data.clear()
    db.node_manager_to_nodes.data.clear()
    
    # C. Load Encodings (Reuse existing)
    enc_dir = 'data/encodings'
    if not os.path.exists(os.path.join(enc_dir, 'visual_train_encodings.npy')):
        print("Encodings not found. Please run scripts/2_generate_encodings.py first.")
        return 0, 0

    v_enc = np.load(os.path.join(enc_dir, 'visual_train_encodings.npy'))
    # Use subset for tuning speed
    v_enc = v_enc[:10000] 
    
    # We don't strictly need labels for the simulation, but we need them for evaluation score
    v_labels = np.load(os.path.join(enc_dir, 'visual_train_labels.npy'))[:10000]
    
    # Audio
    has_audio = os.path.exists(os.path.join(enc_dir, 'audio_train_encodings.npy'))
    if has_audio:
        a_enc = np.load(os.path.join(enc_dir, 'audio_train_encodings.npy'))
        a_labels = np.load(os.path.join(enc_dir, 'audio_train_labels.npy'))
        
    # D. Initialize Brain
    brain = Brain()
    v_cortex = brain.add_cortex('visual', None)
    a_cortex = brain.add_cortex('audio', None)
    
    # E. Run Short Simulation
    for t in range(TUNING_STEPS):
        brain.increment_timestep()
        
        # Random Data
        idx_v = np.random.randint(0, len(v_enc))
        vis_input = v_enc[idx_v]
        label = v_labels[idx_v]
        
        aud_input = None
        if has_audio:
            matching = np.where(a_labels == label)[0]
            if len(matching) > 0:
                idx_a = np.random.choice(matching)
                aud_input = a_enc[idx_a]
            else:
                idx_a = np.random.randint(0, len(a_enc))
                aud_input = a_enc[idx_a]
        
        brain.receive_sensory_input(v_cortex, vis_input)
        if aud_input:
            brain.receive_sensory_input(a_cortex, aud_input)
            
        brain.cleanup()
        brain.create_new_nodes()
        
    # F. Evaluate (Internal Logic)
    # We calculate Unsupervised Accuracy on the subset used
    def get_score(cortex, data, labels):
        if not cortex.node_manager.nodes: return 0.0
        
        cluster_map = {}
        for i in range(len(data)):
            # Quick pass, no learning
            cluster = cortex.receive_sensory_input(data[i], learn=False)
            if cluster:
                if cluster.name not in cluster_map: cluster_map[cluster.name] = []
                cluster_map[cluster.name].append(labels[i])
        
        cluster_label = {}
        for c, lbls in cluster_map.items():
            cluster_label[c] = Counter(lbls).most_common(1)[0][0]
            
        correct = 0
        total = 0
        for i in range(len(data)):
            cluster = cortex.receive_sensory_input(data[i], learn=False)
            if cluster and cluster.name in cluster_label:
                if cluster_label[cluster.name] == labels[i]:
                    correct += 1
            total += 1
        return correct / total if total > 0 else 0

    # Evaluate on a validation subset (last 1000 of training data used)
    v_score = get_score(v_cortex, v_enc[-1000:], v_labels[-1000:])
    a_score = 0
    if has_audio:
        a_score = get_score(a_cortex, a_enc[-1000:], a_labels[-1000:])
        
    print(f"  -> Visual Score: {v_score:.4f} | Audio Score: {a_score:.4f}")
    print(f"  -> Visual Nodes: {len(v_cortex.node_manager.nodes)}")
    
    return v_score, a_score

def main():
    print("=== Hyperparameter Tuning ===")
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    best_score = -1
    best_params = None
    
    for i, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))
        print(f"Trial {i+1}/{len(combinations)}")
        
        v_acc, a_acc = run_tuning_session(current_params)
        
        score = v_acc
        
        if score > best_score:
            best_score = score
            best_params = current_params
            print(f"  *** New record! Score: {best_score:.4f} with params: {best_params} ***")
            
    print("\n=== Tuning Complete ===")
    print(f"Best Score: {best_score}")
    print(f"Best Params: {best_params}")
    
    # Write best_config_found.py
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, 'src', 'best_config_found.py')
    
    with open(config_path, 'w') as f:
        f.write(f"""
class Config:
    # General
    DEVICE = 'cuda'
    EPOCHS = 20
    TRAINING_SET_SIZE = 55000
    
    # Brain Frequency
    BRN_CLEANUP_FREQUENCY = int(TRAINING_SET_SIZE * .25)
    BRN_NEURAL_GROWTH_FREQUENCY = int(TRAINING_SET_SIZE * .3)

    # Cluster
    CLUSTER_REQUIRED_UTILIZATION = int(TRAINING_SET_SIZE * .5)
    CLUSTER_NODE_LEARNING_RATE = 0.02

    # Node
    NODE_REQUIRED_UTILIZATION = int(TRAINING_SET_SIZE * 1.01)
    NODE_POSITION_LEARNING_RATE = 0.04
    NODE_POSITION_MOMENTUM_DECAY = 0.5
    NODE_POSITION_MOMENTUM_ALPHA = 0.0
    NODE_TO_CLUSTER_LEARNING_RATE = 0.05
    NODE_IS_NEW = 25
    NODE_CERTAINTY_AGE_FACTOR = NODE_IS_NEW

    # Node Manager
    NRND_OPTIMIZER_ENABLED = True
    NRND_BUILD_FREQUENCY = int(TRAINING_SET_SIZE / 10)
    NRND_N_TREES = 50
    NRND_SEARCH_K = NRND_N_TREES * 5
    NRND_MAX_AVG_DISTANCE_MOMENTUM = 1e50
    AVG_DISTANCE_MOMENTUM_DECAY = 0.5
    MAX_NODES = 6000
    INITIAL_NODES = 0
    NODE_SPLIT_MAX_CORRELATION_VARIANCE = 5e-3
    NODE_SPLIT_MAX_QTY = max(int(TRAINING_SET_SIZE / 1000), 5)

    # TKBA Settings (TUNED)
    TKBA_VISUAL_SIGMA = {best_params['TKBA_VISUAL_SIGMA']}
    TKBA_VISUAL_VIGILANCE = {best_params['TKBA_VISUAL_VIGILANCE']}
    TKBA_AUDIO_SIGMA = {best_params['TKBA_AUDIO_SIGMA']}
    TKBA_AUDIO_VIGILANCE = {best_params['TKBA_AUDIO_VIGILANCE']}
    
    TKBA_KBR_SIGMA = 1.0
    
    # CDZ
    CE_LEARNING_RATE = 0.02
    CE_IGNORE_GAUSSIAN = True
    CE_CORRELATION_WINDOW_STD = 0.65
    CE_CORRELATION_WINDOW_MAX = 10
    CE_CERTAINTY_AGE_FACTOR = NODE_CERTAINTY_AGE_FACTOR

    # Autoencoder Training
    MNIST_LAYERS = [784, 512, 256, 128]
    FSDD_LAYERS = [4096, 1024, 256, 64]
    AE_LEARNING_RATE = 1e-3
    AE_BATCH_SIZE = 64
    AE_EPOCHS_MNIST = 100
    AE_EPOCHS_FSDD = 150
""")
    print("Saved best configuration to src/best_config_found.py")

if __name__ == "__main__":
    main()
