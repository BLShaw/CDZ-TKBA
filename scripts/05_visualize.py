import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

def visualize():
    print("=== Step 5: Visualization ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    enc_dir = 'data/encodings'
    save_dir = 'data/plots'
    os.makedirs(save_dir, exist_ok=True)

    def plot_tsne(data, labels, title, filename):
        print(f"Generating t-SNE for {title}...")
        # Subsample
        if len(data) > 2000:
            idx = np.random.choice(len(data), 2000, replace=False)
            data = data[idx]
            labels = labels[idx]
            
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(title)
        plt.savefig(filename)
        plt.close()
        print(f"Saved to {filename}")

    # Visual - Test
    if os.path.exists(os.path.join(enc_dir, 'visual_test_encodings.npy')):
        X_v_test = np.load(os.path.join(enc_dir, 'visual_test_encodings.npy'))
        y_v_test = np.load(os.path.join(enc_dir, 'visual_test_labels.npy'))
        plot_tsne(X_v_test, y_v_test, "Visual Encodings (Test) (t-SNE)", os.path.join(save_dir, 'visual_test_tsne.png'))

    # Visual - Train
    if os.path.exists(os.path.join(enc_dir, 'visual_train_encodings.npy')):
        X_v_train = np.load(os.path.join(enc_dir, 'visual_train_encodings.npy'))
        y_v_train = np.load(os.path.join(enc_dir, 'visual_train_labels.npy'))
        plot_tsne(X_v_train, y_v_train, "Visual Encodings (Train) (t-SNE)", os.path.join(save_dir, 'visual_train_tsne.png'))

    # Audio - Test
    if os.path.exists(os.path.join(enc_dir, 'audio_test_encodings.npy')):
        X_a_test = np.load(os.path.join(enc_dir, 'audio_test_encodings.npy'))
        y_a_test = np.load(os.path.join(enc_dir, 'audio_test_labels.npy'))
        plot_tsne(X_a_test, y_a_test, "Audio Encodings (Test) (t-SNE)", os.path.join(save_dir, 'audio_test_tsne.png'))

    # Audio - Train
    if os.path.exists(os.path.join(enc_dir, 'audio_train_encodings.npy')):
        X_a_train = np.load(os.path.join(enc_dir, 'audio_train_encodings.npy'))
        y_a_train = np.load(os.path.join(enc_dir, 'audio_train_labels.npy'))
        plot_tsne(X_a_train, y_a_train, "Audio Encodings (Train) (t-SNE)", os.path.join(save_dir, 'audio_train_tsne.png'))

if __name__ == "__main__":
    visualize()
