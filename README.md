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
```

## Training the Model

The model is trained using the Adam optimizer with a learning rate of `1e-4` (betas: 0.5, 0.999) for 20 epochs by default. A learning rate scheduler (ReduceLROnPlateau) is used to reduce the learning rate if the validation loss plateaus, and early stopping is applied with a patience of 3 epochs.

### Training Steps

1. **Prepare the Dataset:**  
   Organize your curated deblurring datasets in the following structure:
<parent_dir>/<dataset>/<blur>/<train><test>, <sharp>/<train><test>


3. **Update the Parent Directory:**  
In your code (e.g., in `train_deblur.py` or your Jupyter Notebook), update the `parent_dir` variable to point to your local dataset directory. For example:
```python
parent_dir = "./data/DBlur"
```
3. **Run the Training Script or Notebook:**
Save your code as train_deblur.py and run:
```bash
Copy
python train_deblur.py
```
Open your notebook and run the cells sequentially.

## Hyperparameters
1. Epochs: 20
2. Batch Size: 4 (for training), 1 (for evaluation)
3. Learning Rate: 1e-4 (Adam, betas: 0.5, 0.999)
4. Early Stopping Patience: 3 epochs
5. LR Scheduler: ReduceLROnPlateau (factor: 0.5, patience: 2)
6. Base Channel Dimension: 64 (increase to boost capacity)
7. Transformer Settings: 2 layers with 4 heads
8. Loss: L1 Loss + Perceptual Loss (weighted by 0.1)
9. Upsampling: Bicubic interpolation
10. Refinement: Residual refinement block in the decoder
11. Mixed Precision: Enabled via torch.amp

This project builds on state-of-the-art techniques in image deblurring and leverages PyTorch and torchvision for model development and training.
