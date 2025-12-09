# Project Progress Report: Multimodal Decoupled Learning with TKBA

- **Date:** November 28, 2025
- **Project:** Unsupervised Multimodal Grounding (Visual + Audio)
- **Architecture:** Decoupled Hebbian Learning + Topological Kernel Bayesian ART (TKBA)

---

## 1. Project Objective
Building an **unsupervised multimodal learning system** capable of:
1.  Learning to classify visual digits (MNIST) and spoken digits (FSDD) without labels.
2.  "Grounding" knowledge by associating vision and sound (e.g., learning that the image "7" corresponds to the sound "seven").
3.  **Generative Recall**: Demonstrating that seeing an image can trigger the "hallucination" of the corresponding sound, and vice versa.
4.  Using a **decoupled architecture** (separate sensory cortices connected by a Convergence-Divergence Zone or CDZ) rather than a single giant neural network.

---

## 2. Development Roadmap & Logic
### Phase 1: The Foundation
**Goal:** Migrate the outdated TensorFlow 1.X project from FYP-1 to up to date PyTorch with modern programming standards.

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

### Phase 6: Generative Demonstration & "Rich Get Richer" Fix
**Goal:** Prove the "Association" capability (e.g., See "7" -> Say "Seven") and fix the "Super Cluster" dominance.

*   **Issue**: Initial Audio -> Visual generation produced a "generic blob" for most digits.
    *   **Cause**: A single Visual Cluster became a "super-attractor", linking to almost all Audio clusters due to the "Rich Get Richer" effect in Hebbian learning.
*   **Fix 1**: Implemented a **Frequency Penalty** in the CDZ learning rule. Clusters that fire too frequently have their learning rate dynamically throttled (`1 / sqrt(frequency)`).
*   **Fix 2**: Implemented **Competitive Hebbian Inhibition** (Oja's Rule) in the Refinement phase to actively punish incorrect "old" associations when a new, stronger one is found.
*   **Result**: This successfully broke the monopoly of the super-cluster.
*   **Demonstration**: Created `scripts/6_demo_generation.py` which now generates distinct, recognizable visual digits (via centroid decoding) from audio inputs, and vice-versa.
*   **Audio Generation**: Implemented **Griffin-Lim** reconstruction to convert the generated spectrograms back into audible `.wav` files, proving the system can "speak" the digit it sees.

---

## 3. Achievements So Far

We have successfully built and tuned a full-scale unsupervised learning system.

| Modality | Method | Accuracy | Clusters Created | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Visual (MNIST)** | Linear Probe (Supervised) | **92.07%** | N/A | The Autoencoder learned excellent features. |
| **Visual (MNIST)** | **TKBA Brain** (Unsupervised) | **89.96%** | ~3,028 | **Major Achievement**: Achieved nearly 90% unsupervised accuracy. The brain successfully clustered 60,000 digits with high purity. |
| **Audio (FSDD)** | Linear Probe (Supervised) | **89.33%** | N/A | Good latent representation. |
| **Audio (FSDD)** | **TKBA Brain** (Unsupervised) | **92.33%** | ~155 | The brain learned to distinguish spoken digits across different speakers with excellent robustness. |

### Key Takeaway
We moved from a simple prototype to a **mathematically rigorous, kernel-based topological learning system**. The system now behaves like a plastic neural substrate: it grows new neurons when it encounters novel information, refines existing ones when it sees familiar patterns.