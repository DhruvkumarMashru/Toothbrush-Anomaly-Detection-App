# Toothbrush-Anomaly-Detection-App

Overview
This project is designed to detect anomalies in toothbrush images using machine learning. We utilize Teachable Machine to train an image classification model, which is then integrated into a Streamlit app for real-time anomaly detection. The app allows users to upload toothbrush images to detect any defects or issues. Additionally, a bonus feature includes a live camera feed that continuously detects anomalies without requiring image uploads.

Key Features
Anomaly Detection Model: The model is trained using Teachable Machine, a simple no-code tool that enables easy training and export of machine learning models.

Streamlit App: A user-friendly interface that allows image uploads and displays real-time predictions.

Live Camera Feed (Bonus): The app includes a live camera feed that continuously monitors toothbrushes for anomalies.

Anomaly Classification: The model categorizes images as either 'Normal' or 'Defective' based on the trained dataset.

Project Flow
Model Training:

A dataset of toothbrush images is used to train the anomaly detection model in Teachable Machine.

The model is trained to classify images into two categories: 'Normal' (defect-free) and 'Defective' (with issues like cracks, damage, or missing parts).

Model Export:

After training, the model is exported and integrated into a Streamlit application for easy deployment and usage.

Streamlit Application:

Upload Image: Users can upload an image of a toothbrush, and the model will predict if the toothbrush is normal or defective.

Live Camera Feed (Bonus Feature): Users can use their webcam to scan a toothbrush, and the app will predict any anomalies in real-time.

Installation
Requirements
Python 3.x

Streamlit

Keras (for TensorFlow models)

OpenCV (for camera feed)

Teachable Machine model file (model/keras_model.h5)

Model Description
Training Method: The model is trained using Teachable Machine with a custom dataset of toothbrush images. It classifies images into two classes: Normal and Defective.

Data: A collection of normal toothbrush images and images with defects (e.g., cracks, missing bristles) was used to train the model.

Model Type: The model used is a convolutional neural network (CNN), which is well-suited for image classification tasks.

Usage
Upload Image: Select the "Upload Image" option to upload an image of a toothbrush and get a prediction.

The app will classify the image as either "Normal" or "Defective" based on the model's prediction.

Live Camera Feed: Click on the "Start Camera" checkbox to enable the live feed.

