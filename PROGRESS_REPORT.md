# Project Progress Report: Multimodal Decoupled Learning with TKBA

- **Date:** November 28, 2025
- **Project:** Unsupervised Multimodal Grounding (Visual + Audio)
- **Architecture:** Decoupled Hebbian Learning + Topological Kernel Bayesian ART (TKBA)

---

## 1. Project Objective
Building an **unsupervised multimodal learning system** capable of:
1.  Learning to classify visual digits (MNIST) and spoken digits (FSDD) without labels.
2.  "Grounding" knowledge by associating vision and sound (e.g., learning that the image "7" corresponds to the sound "seven").
3.  Using a **decoupled architecture** (separate sensory cortices connected by a Convergence-Divergence Zone or CDZ) rather than a single giant neural network.

---

## 2. Development Roadmap & Logic
### Phase 1: The Foundation
**Goal:** Migrate the outdated TensorFlow project to PyTorch with modern programming standards.

*   **What we did:**
    *   Built the **Brain Class**: The central controller.
    *   Built **Cortices**: Modular processors for Vision and Audio.
    *   Built the **CDZ (Convergence-Divergence Zone)**: The associative engine that links cortices based on temporal co-occurrence ("Cells that fire together, wire together").
    *   Implemented **Autoencoders (AE)**: To compress raw high-dimensional data (images/spectrograms) into compact "encodings" that the brain can process.
*   **Why?** Raw sensory data is too noisy for direct associative learning. Autoencoders provide a stable "latent space" representing the essential features of the data.

### Phase 2: Data Pipeline Engineering
**Goal:** Ensure robust data flow for training.

*   **Challenge:** The audio dataset (FSDD) is raw audio, variable length, and requires download from Git.
*   **Solution:**
    *   Created a `MultimodalDataset` class.
    *   Automated the cloning of the FSDD repository.
    *   Implemented **Spectrogram Generation**: Converted raw audio waveforms into 2D images (spectrograms) to make them compatible with standard CNN/MLP architectures.
    *   Implemented **Silence Trimming**: To remove dead air from recordings, ensuring the model focuses on the digit sound.
    *   **Outcome:** A reliable pipeline that feeds paired (Image, Sound) tuples to the system.

### Phase 3: Optimization & Debugging
**Goal:** Make the system actually learn.

*   **Issue 1: Autoencoder Loss**: Initially, the reconstruction loss was high (~0.6), meaning the brain was looking at "noise".
    *   **Fix**: Switched activation functions to `ReLU` (internal) and `Sigmoid` (output). Normalized audio data to `[0, 1]`.
    *   **Result**: Loss dropped to `0.002`, providing crisp, high-quality inputs to the brain.
*   **Issue 2: Node Growth Stagnation**: The brain stopped learning early.
    *   **Fix**: Adjusted the `TRAINING_SET_SIZE` configuration to match the full dataset, allowing the brain to grow dynamically over long simulations.

### Phase 4: The Major Upgrade (TKBA Integration)
**Goal:** Improve the clustering capability of the Cortices.

*   **The Problem:** The original "Node Manager" used simple Euclidean distance. It struggled to decide when to create a new node vs. update an old one, leading to poor clustering or reliance on fixed parameters.
*   **The Solution:** Integrated **TKBA (Topological Kernel Bayesian Adaptive Resonance Theory)**.
    *   **What is it?** A sophisticated clustering algorithm that uses:
        *   **CIM (Correntropy Induced Metric)**: A kernel-based distance metric robust to outliers.
        *   **Vigilance Parameter**: A dynamic threshold that automatically decides if an input is "novel" (create new node) or "familiar" (update existing node).
*   **Implementation:**
    *   Ported MATLAB logic to Python (`tkba_utils.py`).
    *   Replaced the core `receive_encoding` logic in `NodeManager` with TKBA logic.

### Phase 5: Tuning & Stabilization
**Goal:** Balance the system dynamics.

*   **Challenge 1: The "1 Node" Bug**: The Visual cortex refused to grow, staying at 1 node (11% accuracy).
    *   **Reason**: The Vigilance threshold was too loose; the brain thought *everything* looked like the first node.
*   **Challenge 2: The "Explosion"**: After fixing the above, the node count hit the ceiling (6000) instantly.
    *   **Reason**: Vigilance was too strict; the brain thought *everything* was unique.
*   **Solution: Automated Hyperparameter Tuning**:
    *   We wrote a script (`scripts/6_tune_hyperparams.py`) to grid-search optimal TKBA parameters.
    *   **Discovery**: The visual modality needed a stricter vigilance (0.45) but a slightly smoother kernel (Sigma=0.3) to balance detail with generalization. Audio required high sensitivity (Sigma=0.15).

---

## 3. Final Achievements

We have successfully built and tuned a full-scale unsupervised learning system.

| Modality | Method | Accuracy (Unsupervised) | Clusters Created | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Visual (MNIST)** | Linear Probe | **92.29%** | N/A | The Autoencoder learned excellent features. |
| **Visual (MNIST)** | **TKBA Brain** | **90.11%** | ~3,300 | **Major Achievement**: Broke the 90% unsupervised accuracy barrier. The brain successfully clustered 60,000 digits with high purity. |
| **Audio (FSDD)** | Linear Probe | **89.33%** | N/A | Good latent representation. |
| **Audio (FSDD)** | **TKBA Brain** | **96.67%** | ~223 | The brain learned to distinguish spoken digits across different speakers with near-perfect robustness. |

### Key Takeaway
We moved from a simple prototype to a **mathematically rigorous, kernel-based topological learning system**. The system now behaves like a plastic neural substrate: it grows new neurons when it encounters novel information and refines existing ones when it sees familiar patterns, all without supervision.