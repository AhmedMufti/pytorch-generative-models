# Question 3: Generative Modeling with Variational Autoencoder (VAE)

## Overview
**Problem:** The goal is to generate new, realistic fashion images by learning the underlying distribution of the dataset. Unlike a standard autoencoder which just reconstructs inputs, a VAE allows us to sample new data.

**Dataset:**
- **Source:** [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- **Content:** 70,000 28x28 grayscale images of 10 fashion categories (t-shirt, trouser, pullover, dress, etc.).
- **Split:** 60,000 training images and 10,000 test images.

**Model Check:**
- **Architecture:** Variational Autoencoder (VAE).
- **Why this model?**
  - Standard Autoencoders maps inputs to a fixed vector, which leads to a disjoint latent space (you can't interpolate or sample easily).
  - **VAEs** map inputs to a *probability distribution* (mean and variance) in the latent space. This ensures the latent space is continuous and smooth, making it perfect for **generation**. By sampling a random point from this space, we can decode it into a valid, novel image.

## Approach & Methodology

### 1. Data Preparation
- **Normalization:** Images are loaded as tensors. Since they are grayscale, the input dimension is 1 channel (28x28x1).

### 2. Model Architecture
The VAE introduces a probabilistic twist to the encoder-decoder structure:
- **Encoder:**
  - Flattens the image into a vector.
  - Passes it through fully connected layers.
  - **Crucial Step:** Instead of outputting a single vector $z$, it outputs two vectors:
    1.  $\mu$ (Mean of the latent distribution)
    2.  $\log(\sigma^2)$ (Log Variance of the latent distribution)
- **Reparameterization Trick:**
  - To allow backpropagation through random sampling, we define $z = \mu + \epsilon \cdot \sigma$, where $\epsilon \sim \mathcal{N}(0, 1)$.
  - This allows the network to be differentiable while still being stochastic.
- **Decoder:**
  - Takes the sampled latent vector $z$.
  - Reconstructs it back into the 28x28 image space using fully connected layers.

### 3. Training (The Loss Function)
VAE training involves minimizing a composite loss function:
1.  **Reconstruction Loss (BCE/MSE):** Measures how well the outputs match the inputs (visual fidelity).
2.  **KL Divergence Loss:** Measures how much the learned latent distribution deviates from a standard Normal Distribution $\mathcal{N}(0, 1)$. This acts as a regularizer, forcing the latent space to be well-structured and continuous.

### 4. Generation & Visualization
- **Reconstruction:** We check how well the model reconstructs existing test images.
- **Generation:** We sample random vectors from a standard normal distribution (noise), feed them into the Decoder, and visualize the output. If the model has learned well, these random noise vectors will be transformed into recognizable fashion items.

## Execution Steps (Checklist)

1.  **Environment Setup (Cloud):**
    -   Open [Google Colab](https://colab.research.google.com/).
    -   Set Runtime to **T4 GPU**.
2.  **Upload Files:**
    -   Upload `solution.py`.
    -   *Note:* You do **NOT** need to manualy download the dataset. The script will automatically download Fashion-MNIST.
3.  **Run the Script:**
    -   In a code cell, run: `!python solution.py`
4.  **Monitor Training:**
    -   Watch the console output for "Q3: VAE".
    -   Look for "Average Loss". This number combines both reconstruction error and KL divergence. It should decrease steadily.
5.  **Analyze Results:**
    -   Once finished, refresh the "Files" pane.
    -   **Result 1:** `q3_reconstruction.png`. Compares original images with their reconstructed versions.
    -   **Result 2:** `q3_generated.png`. Shows *new* images created from random noise.
6.  **Iterate (Optional):**
    -   **When to stop:**
        -   Reconstructions should look like the original items (slightly blurrier is normal for VAEs).
        -   Generated images should clearly look like clothing (pants, shirts, shoes), not random static or shapeless blobs.
    -   **How to improve:** If the generated images are unrecognizable, increase `N_EPOCHS` in `solution.py`. VAEs often need a bit more training time to organize the latent space effectively.
