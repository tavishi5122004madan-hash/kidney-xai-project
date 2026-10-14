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
6. Apply Grad-CAM to generate visual explanations.
7. Integrate the model and heatmap into a Gradio web application.
8. Deploy the application using Render.

## 7. Recorded Results

The recorded test accuracy was **98.94% on 379 test images**. The results also showed four Tumor images incorrectly classified as Normal.

These results apply to the evaluated dataset and do not establish clinical reliability or performance on all kidney CT scans.

## 8. Live Application

**Website:** https://kidney-xai-project.onrender.com/

Users can open the web application, upload an image, view the model's prediction, and inspect the Grad-CAM visualization.

## 9. Future Scope

- Evaluate the model on independent datasets.
- Investigate false-negative predictions.
- Improve dataset balance and generalization.
- Assess the quality and consistency of Grad-CAM explanations.
- Add experiment tracking using MLflow.
- Add dataset and pipeline versioning using DVC, if required.
- Improve documentation and testing.

## 10. Limitations

This is an educational and research prototype. The dataset is limited and imbalanced, and the model and its visual explanations require further validation. The system must not be used as a standalone medical diagnostic tool.

## 11. Conclusion

This project demonstrates a workflow combining deep-learning-based kidney CT image classification, Grad-CAM visual explanations, a Gradio interface, and cloud deployment. It aims to make model predictions easier to inspect while identifying areas for further testing and improvement.
