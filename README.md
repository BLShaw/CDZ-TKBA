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
python scripts/1_prepare_data.py
```
*   **Output**: `data/processed/*.npy`, `data/spectrograms/train/*.npy`, `data/spectrograms/test/*.npy`

### 2. Generate Encodings
Trains Autoencoders for both Visual (MNIST) and Audio (FSDD) modalities on the full datasets, then generates latent embeddings (encodings).
```bash
python scripts/2_generate_encodings.py
```
*   **Output**: `data/encodings/*.npy` (Train/Test encodings and labels)

### 3. Run TKBA Learning (Simulation)
Runs the core Brain simulation using **Topological Kernel Bayesian ART (TKBA)**. It initializes Cortices, feeds paired encodings (Visual + Audio), and allows the Brain to learn topology (Nodes/Clusters) and associations (CDZ) dynamically.
```bash
python scripts/3_run_TKBA.py
```
*   **Output**: `data/brain.pkl` (Saved Brain state)

### 4. Evaluate
Performs two types of evaluation:
1.  **Supervised**: Linear probe (Logistic Regression) on raw encodings to verify embedding quality.
2.  **Unsupervised**: Passes the full test set through the trained Brain, maps activated clusters to labels, and calculates clustering accuracy.
```bash
python scripts/4_evaluate.py
```

### 5. Visualize
Generates t-SNE plots to visualize the latent spaces of the Visual and Audio modalities (Train and Test splits).
```bash
python scripts/5_visualize.py
```
*   **Output**: `data/plots/visual_train_tsne.png`, `data/plots/visual_test_tsne.png`, `data/plots/audio_train_tsne.png`, `data/plots/audio_test_tsne.png`

## Optional: Hyperparameter Tuning
If you wish to re-optimize the TKBA parameters for your specific environment:
```bash
python scripts/6_tune_hyperparams.py
```
*   **Output**: `src/best_config_found.py` (which you can manually merge into `src/config.py`)

## Troubleshooting

*   **Git Error**: If you see an error about `git`, ensure Git is installed and added to your system PATH. Alternatively, verify you can run `git --version` in your terminal.
*   **Download Error**: If datasets fail to download, manually delete the `data` folder and try again, or verify your internet connection.
*   **Memory Issues**: If you run out of RAM during Step 2 or 3, try reducing `BATCH_SIZE` in `src/config.py`.