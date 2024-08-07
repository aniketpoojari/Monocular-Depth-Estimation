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

Image denoising is a crucial task in image processing and computer vision, as noise can degrade image quality and hinder further analysis. In this project, we will explore the use of an autoencoder neural network for image denoising.

An autoencoder is a type of neural network that is trained to reconstruct its input. It consists of an encoder that maps the input to a lower-dimensional latent space, and a decoder that maps the latent space back to the original input space. By training the autoencoder to reconstruct noisy images, we can effectively teach it to remove the noise and produce clean, denoised images.

Our autoencoder will be implemented using PyTorch, a popular open-source machine learning library. We will define the encoder and decoder architectures using sequential blocks of convolutional or transpose convolutional layers, followed by batch normalization and ReLU activation.

To train the autoencoder, we will use the Adam optimizer and the mean squared error loss function. We will also log the training metrics and the final model using MLflow, an open-source platform for managing machine learning experiments.

## 🌟 Features

- [Data Loader](src/data_loader.py)
- [Model](src/model.py)
- [Training](src/training.py)
- [Best Model Selection](src/log_production_model.py)

## 🛠️ Requirements

- yaml
- torch
- torchvision
- os
- pillow
- numpy
- visdom
- mlflow

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
