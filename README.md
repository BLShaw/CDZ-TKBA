# Multimodal Brain Project - Execution Guide

This project simulates a decoupled multimodal learning architecture using MNIST (Visual) and Free Spoken Digit Dataset (Audio). The pipeline is split into distinct stages for data preparation, encoding generation, simulation, evaluation, and visualization.

## Prerequisites

1.  **Python 3.8+** installed.
2.  **Git** installed (required for downloading FSDD).
3.  **CUDA-capable GPU** (Recommended for faster training and simulation).

## Setup

1.  **Install Dependencies**:
    Open a terminal in the project root and run:
    ```bash
    pip install -r requirements.txt
    ```

## Execution Pipeline

Run the following scripts in sequence from the project root directory:

### 1. Prepare Data
Downloads datasets (MNIST, FSDD), trims audio silence, generates spectrograms, and saves processed data.
```bash
python scripts/01_prepare_data.py
```
*   **Output**: `data/processed/*.npy`, `data/spectrograms/train/*.npy`, `data/spectrograms/test/*.npy`

### 2. Generate Encodings
Trains Autoencoders for both Visual (MNIST) and Audio (FSDD) modalities on the full datasets, then generates latent embeddings (encodings).
```bash
python scripts/02_generate_encodings.py
```
*   **Output**: `data/encodings/*.npy` (Train/Test encodings and labels)

### 3. Run TKBA Learning (Simulation)
Runs the core Brain simulation using **Topological Kernel Bayesian ART (TKBA)**. It initializes Cortices, feeds paired encodings (Visual + Audio), and allows the Brain to learn topology (Nodes/Clusters) and associations (CDZ) dynamically.
```bash
python scripts/03_run_TKBA.py
```
*   **Output**: `data/brain.pkl` (Saved Brain state)

### 4. Evaluate
Performs two types of evaluation:
1.  **Supervised**: Linear probe (Logistic Regression) on raw encodings to verify embedding quality.
2.  **Unsupervised**: Passes the full test set through the trained Brain, maps activated clusters to labels, and calculates clustering accuracy.
```bash
python scripts/04_evaluate.py
```

### 5. Visualize
Generates t-SNE plots to visualize the latent spaces of the Visual and Audio modalities (Train and Test splits).
```bash
python scripts/05_visualize.py
```
*   **Output**: `data/plots/visual_train_tsne.png`, `data/plots/visual_test_tsne.png`, `data/plots/audio_train_tsne.png`, `data/plots/audio_test_tsne.png`

### 6. Hyperparameter Tuning
Optimizes the TKBA parameters (Sigma, Vigilance) for your specific environment to maximize unsupervised clustering accuracy.
```bash
python scripts/06_tune_hyperparams.py
```
*   **Output**: `src/best_config_found.py` (which you can manually merge into `src/config.py`)

### 7. Generative Demonstration
Demonstrates the system's "Generative Recall" capability. It takes an input image/sound, finds the associated cluster in the other modality via the CDZ, and reconstructs the output (Image or Audio/Spectrogram).
```bash
python scripts/07_demo_generation.py
```
*   **Output**: `data/demo_outputs/*.png`, `data/demo_outputs/*.wav`

### 8. Refine Associations (Highly Recommended)
Runs a "Freeze & Associate" phase using **Competitive Hebbian Learning**. It freezes the learned nodes and re-trains the associations with a very low learning rate to fix any potential initial label misalignments.
```bash
python scripts/08_refine_associations.py
```
*   **Output**: Updates `data/brain.pkl` with refined associations.

## Troubleshooting

*   **Git Error**: If you see an error about `git`, ensure Git is installed and added to your system PATH. Alternatively, verify you can run `git --version` in your terminal.
*   **Download Error**: If datasets fail to download, manually delete the `data` folder and try again, or verify your internet connection.
*   **Memory Issues**: If you run out of RAM during Step 2 or 3, try reducing `BATCH_SIZE` in `src/config.py`.
