# Question 2: Image Denoising using Denoising Autoencoder

## Overview
**Problem:** The objective is to restore clean images from noisy versions. This simulates real-world scenarios where images are corrupted by sensor noise or transmission errors.

**Dataset:**
- **Source:** [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Content:** 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, etc.).
- **Split:** 50,000 training images and 10,000 test images.

**Model Check:**
- **Architecture:** Convolutional Denoising Autoencoder (DAE).
- **Why this model?**
  - **Convolutional Neural Networks (CNNs)** are tailored for image data, capturing spatial hierarchies (edges, textures, objects).
  - **Autoencoders** are unsupervised learners designed to capture the most important features of data. By forcing the model to reconstruct the *clean* image from a *noisy* input, it learns to ignore the noise and focus on the underlying structure.

## Approach & Methodology

### 1. Data Preparation & Noise Injection
- **Normalization:** Images are converted to tensors with values in [0, 1].
- **Noise Injection:** We artificially corrupt the clean images before feeding them to the network.
  - **Gaussian Noise:** Adds random values from a normal distribution.
  - **Salt-and-Pepper Noise:** Randomly sets pixels to black (0) or white (1).

### 2. Model Architecture
- **Encoder (Compression):**
  - Consists of Convolutional layers (`Conv2d`) followed by ReLU activations and Max Pooling.
  - Compresses the 32x32x3 input image into a smaller, dense latent representation (feature map).
- **Decoder (Reconstruction):**
  - Uses Transposed Convolutional layers (`ConvTranspose2d`) to upsample the latent representation back to the original dimension (32x32x3).
  - The final layer uses a Sigmoid activation to ensure pixel values remain between 0 and 1.

### 3. Training
- **Input:** Noisy Image.
- **Target:** Original Clean Image.
- **Loss Function:** Mean Squared Error (MSE) Loss. This penalizes valid pixel-wise differences between the reconstructed image and the original clean image.
- **Process:** The network minimizes the MSE, effectively learning a function that maps "noisy" pixel patterns back to their "clean" counterparts.

### 4. Evaluation
- **Visual Inspection:** We plot the Original, Noisy, and Reconstructed images side-by-side to visually verify the denoising performance.
- **Quantitative Metrics:** MSE Loss on the test set.

## Execution Steps (Checklist)

1.  **Environment Setup (Cloud):**
    -   Open [Google Colab](https://colab.research.google.com/).
    -   Create a new notebook.
    -   **Important:** Set Runtime to **T4 GPU**.
2.  **Upload Files:**
    -   Upload `solution.py`.
    -   *Note:* You do **NOT** need to manually download the dataset. The script will automatically download the CIFAR-10 dataset to the cloud environment.
3.  **Run the Script:**
    -   In a code cell, run: `!python solution.py`
4.  **Monitor Training:**
    -   Watch the console output for "Q2: Denoising Autoencoder".
    -   The Training Loss should start around 0.1-0.2 and decrease significantly.
5.  **Analyze Results:**
    -   Once finished, refresh the "Files" pane in Colab/Kaggle.
    -   Look for an image file named **`q2_results.png`**.
    -   Double-click to open validation image.
    -   **Check:** The "Reconstructed" row should look much clearer than the "Noisy" row and resemble the "Original".
6.  **Iterate (Optional):**
    -   **When to stop:** If the "Reconstructed" images are still very grainy or look like gray blobs.
    -   **How to improve:** Edit `solution.py` and increase `N_EPOCHS` (e.g., to 10 or 20). The standard 2 epochs is minimal for demonstration.
