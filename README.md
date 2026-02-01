# Facial Expression Recognition using PyTorch

## 📌 Project Overview
This project implements a **Facial Expression Recognition system** using **Deep Learning and Transfer Learning**.  
The model classifies human facial images into different emotional categories using a pretrained **EfficientNet-B0** model in **PyTorch**.

The project demonstrates the complete **machine learning pipeline** — from data preprocessing to model training, validation, and performance evaluation.

---

## 🎯 Emotion Classes
The model predicts the following **7 facial expressions**:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## 🚀 Key Features
- Transfer Learning with **EfficientNet-B0**
- Image augmentation for better generalization
- GPU (CUDA) support
- Automatic best-model checkpoint saving
- Training & validation accuracy tracking
- Clean and modular PyTorch implementation

---

## 📂 Dataset
**Dataset Source:**  
[Facial Expression Recognition](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
## 🧠 Model Architecture

- **Base Model:** EfficientNet-B0 (pretrained on ImageNet)
- **Framework:** PyTorch
- **Output Layer:** Modified fully connected layer for 7 emotion classes
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam

---

## 🔄 Input–Output Methodology

### 🔹 Input
- RGB facial images
- Images resized and normalized
- Training data augmented using:
  - Random horizontal flip
  - Random rotation
- Validation data kept unaugmented for fair evaluation

### 🔹 Processing Pipeline
1. Images are loaded using PyTorch `ImageFolder`
2. Data is passed in batches using `DataLoader`
3. Forward pass is performed through EfficientNet-B0
4. Loss is calculated using cross-entropy loss
5. Backpropagation is performed
6. Model weights are updated using Adam optimizer

### 🔹 Output
- Predicted facial expression label
- Class probability scores
- Epoch-wise loss and accuracy
- Best model saved automatically based on validation loss

---

## ⚙️ Training Configuration

| Parameter      | Value |
|---------------|-------|
| Batch Size    | 32    |
| Learning Rate | 0.001 |
| Epochs        | 15    |
| Optimizer     | Adam  |
| Device        | GPU (CUDA) |

---

## 📈 Training Results Summary

- **Initial Training Accuracy:** ~37%
- **Final Training Accuracy:** ~69%
- **Best Validation Accuracy:** ~64%
- Training loss consistently decreased across epochs
- Best model checkpoint saved whenever validation loss improved

These results demonstrate effective learning and reasonable generalization using transfer learning.

---

## 💾 Model Checkpointing

- Model is automatically saved whenever validation loss improves
- Ensures the best-performing model is preserved for deployment or inference



