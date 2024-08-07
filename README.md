# 🚀 Project Name

Monocular Depth Estimation

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## 📄 Introduction

Monocular Depth Estimation is a critical task in computer vision, enabling the inference of scene depth information from a single RGB image. This capability is essential for applications such as 3D reconstruction, autonomous driving, and augmented reality. In this project, we will explore the use of a Vision Transformer (ViT) combined with convolutional decoder blocks for monocular depth estimation.

Monocular depth estimation involves training a neural network to predict a dense depth map from a single image. The network learns to interpret various visual cues present in the image that correspond to depth information, such as object size, texture gradients, and occlusion boundaries.

Our model will leverage a Vision Transformer (ViT) as the encoder. The ViT is well-suited for capturing global context and long-range dependencies within the image, which are crucial for accurate depth estimation. The encoded features will then be passed to a series of convolutional decoder blocks, which will upsample the features and generate the corresponding depth map.

The model will be implemented using PyTorch, a popular open-source machine learning library. The ViT encoder will extract high-level features from the input image, while the convolutional decoder blocks will progressively refine these features to produce a high-resolution depth map.

To train the depth estimation model, we will use the Adam optimizer and the root mean squared error loss function, which measures the difference between the predicted depth map and the ground truth depth. We will also log the training metrics and the final model using MLflow, an open-source platform for managing machine learning experiments.

Based on the paper: https://arxiv.org/pdf/2103.13413v1

## 🌟 Features

- [Data Loader](src/data_loader.py)
- [Model](src/model.py)
- [Training](src/training.py)
- [Best Model Selection](src/log_production_model.py)

## 🛠️ Requirements

- Pillow
- torch
- torchvision
- pandas
- mlflow
- timm
- PyYAML

## 🚚 Installation

```bash
# Clone the repository
git clone https://github.com/aniketpoojari/Monocular-Depth-Estimation.git
# Change directory
cd Monocular-Depth-Estimation

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Run mlflow server in the background before running the pipeline
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0
```

## 📊 Data

Downloaded from the Cityscapes Dataset Disparity Maps - https://www.cityscapes-dataset.com/downloads/

## 🤖 Model Training

```bash
# Change values in the params.yaml file for testing different parameters
# Train the model using this command
dvc repro
```

## 📈 Evaluation

- Root Mean Squared Error in cm is used to evaluate the model

## 🎉 Results

- Go to MLFLOW server to look at the results.
- saved_models folder will contain the final model after the pipeline is executed using MLFlow
