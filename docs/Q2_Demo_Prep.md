# Q2 Demo Prep — Image Denoising (Denoising Autoencoder)

## What is Q2 About?
We built a **Convolutional Denoising Autoencoder (DAE)** to clean up corrupted images. 
An Autoencoder is a neural network designed to copy its input to its output, but it is forced to pass the data through a "bottleneck" (a compressed representation). For denoising, we intentionally feed it a noisy image and task it with reconstructing the original, clean image.

- **Dataset:** CIFAR-10 (60,000 color images: airplanes, cars, birds, cats, etc., 32x32 pixels)
- **Model Type:** Convolutional Neural Network (CNN) Autoencoder
- **Noise Types Tested:** Gaussian Noise and Salt-and-Pepper Noise

---

## The Big Picture (How DAE Works in Our Code)

```
Clean Image + Artificial Noise → [NOISY INPUT] → [ENCODER] → [BOTTLENECK] → [DECODER] → [CLEAN PREDICTION]
```

### 1. ENCODER (Compresses the image)
- Takes a noisy 32x32 RGB image (3 channels).
- Uses Convolutional layers (like image filters) and Max Pooling (shrinks the image).
- Result: The image is compressed down to a small **8x8 representation** with 64 channels (the "Bottleneck").

### 2. DECODER (Reconstructs the image)
- Takes the compressed 8x8 representation.
- Uses Transpose Convolutional layers (also called Deconvolutions) to "upsample" or enlarge the image back to its original size.
- A final `Sigmoid` activation ensures all pixel values are between 0 and 1.
- Result: A clean 32x32 RGB image.

**Key Idea:** The network must learn the underlying structure of the objects in CIFAR-10 to successfully reconstruct them from the compressed bottleneck. Because noise is random and unpredictable, the bottleneck cannot store the noise. Therefore, the decoder naturally outputs a smooth, noise-free image.

---

## Step-by-Step: What the Code Does (In Order)

### Task 1: Dataset Loading and Preprocessing
- Downloaded CIFAR-10 dataset.
- Transformed images to PyTorch tensors (values scaled from 0-255 to 0.0-1.0).
- Created a random Train (90%) / Validation (10%) split.
- **Batching:** Processed images in batches of 128 for GPU efficiency.

### Task 2: Injecting Artificial Noise
We added two types of noise (with a severity `factor=0.3`):
1. **Gaussian Noise:** Simulates sensor static in low light.
   - We added random values from a normal distribution to every single pixel.
   - Used `torch.clamp` to ensure values do not go below 0 or above 1.
2. **Salt-and-Pepper Noise:** Simulates dead sensor pixels or data transmission errors.
   - Randomly forced 15% of pixels to be completely black (pepper=0) and 15% to be completely white (salt=1).

**If asked:** "We synthetically generated noisy images during training by taking clean CIFAR-10 images and overlaying either Gaussian noise or Salt-and-Pepper noise on the fly inside the training loop."

### Task 3: Building the DAE Architecture
- Symmetric design.
- **Encoder:** 2 Conv layers, Batch Normalization, ReLU activation, 2 MaxPool steps.
- **Decoder:** 2 ConvTranspose layers, Batch Normalization, ReLU activation, final Sigmoid.
- **Total Parameters:** Approximately 88k parameters.

**If asked:** "Batch Normalization was included after each convolutional layer to stabilize training, and we used Max Pooling in the encoder to reduce spatial dimensions, paired with ConvTranspose in the decoder to recover them."

### Task 4 & 5: Training and Evaluation Metrics
- **Optimizer:** Adam with learning rate `0.001`.
- **Loss Function:** `MSELoss` (Mean Squared Error). It compares the reconstructed image pixel-by-pixel against the original clean image.
- Trained separate models for 15 epochs: one for Gaussian noise, one for Salt-and-Pepper.

We used three evaluation metrics:
1. **MSE (Mean Squared Error):** Lower is better. The raw pixel difference.
2. **PSNR (Peak Signal-to-Noise Ratio):** Higher is better. A standard image quality metric measured in decibels (dB). Above 20 dB is generally acceptable.
3. **SSIM (Structural Similarity Index):** Higher is better (max is 1.0). Measures perceived image quality by looking at underlying structure rather than just raw pixel values.

**Results on Test Set (Noise Factor = 0.3):**
| Noise Type | PSNR (dB) | SSIM |
|------------|----------|------|
| Gaussian | 21.74 | 0.944 |
| Salt-Pepper | 23.46 | 0.964 |

**If asked:** "The model performed noticeably better on Salt-and-Pepper noise. This makes sense because S&P noise completely destroys random individual pixels while leaving the surrounding pixels intact, making it easier for convolutions to fill in the gaps. Gaussian noise corrupts every single pixel simultaneously, making the exact original color harder to recover."

### Task 6: Visualizations and Experimental Study
We tracked training vs validation loss.
- Both train and val MSE decreased smoothly, showing no major overfitting.
- We generated side-by-side plots: Original vs. Noisy vs. Reconstructed.
- We varied the noise factor {0.1, 0.3, 0.5, 0.7} and bottleneck sizes {16, 32, 64, 128} to observe how the network degrades under extreme conditions.

---

## Quick Demo Q&A Cheat Sheet

**Q: What is a Denoising Autoencoder?**
A: It is an autoencoder explicitly trained to map a corrupted, noisy image back to its original clean version by forcing it through a lower-dimensional bottleneck.

**Q: Why use MSE Loss instead of Cross Entropy?**
A: Cross Entropy is for classification (picking a discrete class/word). MSE is for regression. We are predicting continuous pixel color intensity values between 0.0 and 1.0, so Mean Squared Error measures the exact distance in color space.

**Q: Why do the reconstructed images look slightly blurry?**
A: This is a known limitation of using MSE as a loss function. When the model is unsure about a fine detail (like the exact texture of a dog's fur), MSE encourages it to "hedge its bets" by predicting the mathematical average of all possible colors. The average of sharp details results in a blurry image.

**Q: Did the model overfit?**
A: No. Unlike Q1, the validation loss curves for both Gaussian and S&P noise followed the training curves closely and did not diverge upwards.

**Q: What is the bottleneck, and why is it important?**
A: The bottleneck connects the encoder to the decoder. It is the most compressed representation of the image (in our case, 64 channels of 8x8 pixels). It forces the network to learn only the most important semantic features of the image, discarding random noise which cannot be compressed easily.

**Q: How does the network upsample in the decoder?**
A: We used `ConvTranspose2d` layers. These act like "reverse convolutions," learning how to take a single pixel and expand it into a larger 2x2 spatial area, allowing us to reconstruct the 32x32 image from the 8x8 bottleneck.

**Q: Why did it perform better on Salt and Pepper noise?**
A: Salt-and-Pepper noise sets pixels to completely black or white, but leaves adjacent pixels perfect. Convolutional filters are extremely good at looking at perfect surrounding pixels and guessing the missing pixel. Gaussian noise alters *every* pixel by a random amount, leaving the network with no "clean" local reference points.
