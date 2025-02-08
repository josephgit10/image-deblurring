# image_deblurring
Image Deblurring using "A Curated List of Image Deblurring Datasets"

# HybridDeblurNet: A Hybrid CNN–Transformer Model for Image Deblurring

This repository contains an image deblurring method that uses a hybrid architecture combining Convolutional Neural Networks (CNNs) and Transformer modules. The model—called **HybridDeblurNet**—is designed to remove blur from images (e.g., due to camera shake or motion) by learning both local features (via CNNs) and long-range dependencies (via Transformer blocks).

## Features

- **Hybrid Architecture:**  
  Combines a CNN encoder with Transformer encoder and decoder blocks, along with a refinement (residual) block for enhanced feature processing.

- **Loss Functions:**  
  Uses a combination of L1 reconstruction loss and a perceptual loss (based on VGG16 features) to generate sharper, more realistic outputs. An optional gradient (edge-aware) loss can also be added.

- **Upsampling Enhancements:**  
  Uses bicubic interpolation for upsampling to improve image quality.

- **Mixed Precision Training:**  
  Leverages PyTorch AMP (`torch.amp.autocast("cuda")` and GradScaler) for faster training and reduced memory usage.

- **Additional Visualizations:**  
  Provides functions to save evaluation metrics (PSNR, SSIM distributions and scatter plots) and to save sample outputs into separate folders.

## Repository Structure

- `train_deblur.py`: Main training script that loads data, builds the model, trains, evaluates, and visualizes results.
- `README.md`: This file.
- `requirements.txt`: List of required packages.
- `sample_outputs/`: (Created at runtime) Contains subfolders with sample images:
  - `blurred/`
  - `deblurred/`
  - `sharp/`
- Additional output files such as:
  - `training_loss_curve.png`
  - `psnr_distribution.png`
  - `ssim_distribution.png`
  - `psnr_vs_ssim.png`
  - `evaluation_metrics.txt`
  - `100_samples.png` (or split files, if configured)

## Setup

### 1. Environment Setup (Local)

It is recommended to use Python 3.7+ and create a virtual environment:

```bash
# Create and activate a virtual environment (Linux/Mac)
python -m venv deblur-env
source deblur-env/bin/activate

# On Windows:
python -m venv deblur-env
deblur-env\Scripts\activate
