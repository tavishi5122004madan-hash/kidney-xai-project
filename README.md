# Explainable AI-Based Kidney Tumor Classification

## 1. Project Overview

This project is a deep-learning-based system that classifies kidney CT scan images into two categories: **Normal** and **Tumor**. It uses Explainable Artificial Intelligence (XAI) to provide a visual explanation of the model's predictions.

The system uses the EfficientNetB0 architecture for image classification and Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight image regions that influenced a prediction.

## 2. Problem Statement

Deep-learning models can classify medical images, but their predictions are not always easy to interpret. This project aims to combine kidney image classification with visual explanations to make model outputs easier to inspect and understand.

## 3. Objectives

- Classify kidney CT scan images into Normal and Tumor categories.
- Apply transfer learning using EfficientNetB0.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Generate Grad-CAM heatmaps to visualize influential image regions.
- Develop a web interface for image upload and prediction.
- Deploy the application online for demonstration.

## 4. Technologies Used

- **Programming language:** Python
- **Deep learning:** TensorFlow and Keras
- **Model architecture:** EfficientNetB0
- **Explainable AI:** Grad-CAM
- **Web interface:** Gradio
- **Development environment:** Google Colab
- **Version control:** Git and GitHub
- **Deployment:** Render

## 5. Dataset

The dataset contains 2,523 images:

- Normal: 240 images
- Tumor: 2,283 images

The images were divided into training, validation, and test sets. The dataset is imbalanced, so class-specific metrics are important when interpreting results.

## 6. Methodology

1. Collect and organize the kidney CT scan images.
2. Prepare the dataset and split it into training, validation, and test sets.
3. Use the pre-trained EfficientNetB0 model for feature extraction.
4. Train the classification layers for the two image categories.
5. Evaluate performance on the test set.
6. Apply Grad-CAM to
