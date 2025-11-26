import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter

def evaluate():
    print("=== Step 4: Evaluation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    enc_dir = 'data/encodings'
    
    # Load Data
    X_v_train = np.load(os.path.join(enc_dir, 'visual_train_encodings.npy'))
    y_v_train = np.load(os.path.join(enc_dir, 'visual_train_labels.npy'))
    X_v_test = np.load(os.path.join(enc_dir, 'visual_test_encodings.npy'))
    y_v_test = np.load(os.path.join(enc_dir, 'visual_test_labels.npy'))
    
    has_audio = os.path.exists(os.path.join(enc_dir, 'audio_train_encodings.npy'))
    
    # 1. Supervised Evaluation (Linear Classifier on Encodings)
    print("\n--- Supervised Evaluation (Linear Probe on Encodings) ---")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_v_train, y_v_train)
    preds = clf.predict(X_v_test)
    print(f"Visual Modality Accuracy: {accuracy_score(y_v_test, preds):.4f}")
    
    if has_audio:
        X_a_train = np.load(os.path.join(enc_dir, 'audio_train_encodings.npy'))
        y_a_train = np.load(os.path.join(enc_dir, 'audio_train_labels.npy'))
        X_a_test = np.load(os.path.join(enc_dir, 'audio_test_encodings.npy'))
        y_a_test = np.load(os.path.join(enc_dir, 'audio_test_labels.npy'))
        
        clf_a = LogisticRegression(max_iter=1000)
        clf_a.fit(X_a_train, y_a_train)
        preds_a = clf_a.predict(X_a_test)
        print(f"Audio Modality Accuracy: {accuracy_score(y_a_test, preds_a):.4f}")

    # 2. Unsupervised Evaluation (Brain Clusters)
    print("\n--- Unsupervised Evaluation (Brain Clusters) ---")
    if not os.path.exists('data/brain.pkl'):
        print("Brain state not found. Skipping.")
        return

    # Restore global DB reference used by pickled nodes
    from src.core.database import db
    
    with open('data/brain.pkl', 'rb') as f:
        data_dump = pickle.load(f)
        
    if isinstance(data_dump, tuple) and len(data_dump) == 2:
        brain, db_state = data_dump
        # Restore DB state
        db.nodes.data = db_state['nodes']
        db.clusters.data = db_state['clusters']
        db.nodes_to_clusters.data = db_state['nodes_to_clusters']
        db.clusters_to_nodes.data = db_state['clusters_to_nodes']
        db.node_manager_to_nodes.data = db_state['node_manager_to_nodes']
        print("Database state restored.")
    else:
        # Legacy or direct brain dump (won't work if DB is empty)
        brain = data_dump
        print("Warning: Only Brain loaded. Database might be empty (Nodes won't work).")

    def eval_unsupervised(cortex, encodings, labels):
        if not cortex.node_manager.nodes:
            print(f"{cortex.name}: No nodes.")
            return
            
        # Map each cluster to a label based on frequency
        cluster_map = {} # cluster_name -> list of labels
        
        # Pass all data through cortex to see which cluster activates
        
        print(f"Mapping {cortex.name} clusters...")
        preds = []
        
        # Iterate.
        for i in range(len(encodings)):
            # receive_sensory_input returns the cluster object
            # set learn=False
            cluster = cortex.receive_sensory_input(encodings[i], learn=False)
            if cluster:
                if cluster.name not in cluster_map:
                    cluster_map[cluster.name] = []
                cluster_map[cluster.name].append(labels[i])
        
        # Determine dominant label for each cluster
        cluster_label_mapping = {}
        for c_name, lbls in cluster_map.items():
            most_common = Counter(lbls).most_common(1)[0][0]
            cluster_label_mapping[c_name] = most_common
            
        # Calculate accuracy
        correct = 0
        total = 0
        for i in range(len(encodings)):
            cluster = cortex.receive_sensory_input(encodings[i], learn=False)
            if cluster and cluster.name in cluster_label_mapping:
                pred = cluster_label_mapping[cluster.name]
                if pred == labels[i]:
                    correct += 1
            total += 1
            
        print(f"{cortex.name} Unsupervised Accuracy: {correct/total:.4f} (Clusters: {len(cluster_label_mapping)})")

    # Evaluate Visual Cortex
    v_cortex = brain.get_cortex('visual')
    # Use Test Set for evaluation
    eval_unsupervised(v_cortex, X_v_test, y_v_test) 

    if has_audio:
        a_cortex = brain.get_cortex('audio')
        X_a_test = np.load(os.path.join(enc_dir, 'audio_test_encodings.npy'))
        y_a_test = np.load(os.path.join(enc_dir, 'audio_test_labels.npy'))
        eval_unsupervised(a_cortex, X_a_test, y_a_test)

if __name__ == "__main__":
    evaluate()
