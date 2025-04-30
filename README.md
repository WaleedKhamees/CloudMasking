# Satellite Project: Cloud Masking

This project implements a cloud masking system using deep learning, specifically designed to process satellite images in TIFF format and identify cloud regions with high accuracy.

## Project Overview

The project follows a typical deep learning pipeline from data preprocessing to inference, using a modified U-Net architecture enhanced with SqueezeNet-inspired Fire modules.

---

## 📦 Preprocessing

The preprocessing pipeline includes:

- Loading and organizing satellite images and their corresponding cloud masks from TIFF files.
- Converting GDAL datasets to NumPy arrays.
- Splitting the dataset:
  - Training: 60%
  - Validation: 20%
  - Test: 20%
- Data augmentation: random horizontal/vertical flips and 45° rotations.
- Efficient batch processing using PyTorch’s `DataLoader`.

---

## 📊 Exploratory Data Analysis

- Dataset size: 10,573 satellite images with cloud masks.
- Image specs:
  - Format: TIFF
  - Spectral bands: 4
  - Dimensions: 4x512x512
- Mask specs: binary, 1x512x512
- Data distribution:
  - Train: 6,343
  - Validation: 2,115
  - Test: 2,115
- Shuffling with random seed `42` for reproducibility.

---

## 🧠 Model Architecture

Modified U-Net with the following features:

- Encoder-decoder with skip connections.
- FireModule (squeeze-expand architecture) as initial layer.
- 4 downsampling and 4 upsampling levels.
- Batch normalization and dropout (p=0.5).
- Final sigmoid activation for binary output.
- Input: 4 channels | Output: 1 channel.

### 🔧 Key Components

- **FireModule**: 1x1 conv squeeze → 1x1 and 3x3 conv expansions.
- **ConvBlock**: Two conv layers with optional batch norm & dropout.
- Transposed convolutions for upsampling.

---

## ⚙️ Training Configuration

- Batch size: 16
- Learning rate: 0.001
- Epochs: 20
- Loss: Binary Cross Entropy Loss (BCELoss)
- Optimizer: SGD
- Dropout: 0.5
- Hardware: NVIDIA T4 GPU (Kaggle)
- Early stopping based on validation Dice coefficient.

---

## 📈 Training Results

- Initial training loss: 0.6924
- Final training loss: 0.5858
- Best validation Dice coefficient: 0.8562 (epoch 19)
- Training duration: ~4 hours

---

## ✅ Evaluation Metrics

- Test Dice Coefficient: 0.8473
- Model size: 7.7M parameters
- Operations: 92.6 billion
- Dice improved from 0.8027 (epoch 1) → 0.8562 (epoch 19)

---

## 🚀 Inference Pipeline

- Load model and evaluate on test data.
- Threshold predictions at 0.5.
- Batch processing of test set.
- Generate submission CSV file.

---

## 🔮 Future Work

- Explore alternate loss functions (Dice loss, Focal loss).
- Test advanced architectures (Attention U-Net, ResUNet).

