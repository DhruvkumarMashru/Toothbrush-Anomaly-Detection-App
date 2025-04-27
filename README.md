# Toothbrush Anomaly Detection

This project is designed to detect anomalies in toothbrush images using machine learning. It leverages a model trained with **Teachable Machine** and integrates it into a **Streamlit** application for real-time detection.


## Key Features

- **Anomaly Detection Model**:  
  Trained using Google's Teachable Machine — a no-code platform — for easy creation and export of machine learning models.
  
- **Streamlit Web App**:  
  A user-friendly interface allowing users to upload toothbrush images and receive real-time predictions.

- **Live Camera Feed (Bonus Feature)**:  
  Integrated webcam functionality to continuously monitor toothbrushes for anomalies in real-time.

- **Anomaly Classification**:  
  The model classifies toothbrush images as either **'Normal'** or **'Defective'** based on visual defects like cracks, missing parts, or damaged bristles.

---

## Project Flow

### 1. Model Training
- A dataset of toothbrush images is used to train the model.
- Images are classified into two categories:  
  - **Normal**: Defect-free toothbrushes.  
  - **Defective**: Toothbrushes with visible defects (e.g., cracks, missing bristles, damaged parts).

### 2. Model Export
- After training, the model is exported from Teachable Machine.
- The exported model (`keras_model.h5`) is integrated into a Streamlit application for deployment.

### 3. Streamlit Application
- **Upload Image**:  
  Users can upload a toothbrush image, and the model will predict its status.

- **Live Camera Feed**:  
  Users can activate their webcam to scan toothbrushes continuously and detect anomalies in real-time.

---

## Installation Requirements

Make sure you have the following installed:

- Python 3.x
- [Streamlit](https://streamlit.io/)
- Keras (with TensorFlow backend)
- OpenCV (for webcam integration)
- Exported Teachable Machine model (`model/keras_model.h5`)

You can install the required Python libraries with:

```bash
pip install streamlit keras tensorflow opencv-python
```

---

## Model Description

- **Training Method**:  
  Trained using **Teachable Machine** with a custom dataset of toothbrush images.

- **Data**:  
  Two categories — **Normal** and **Defective** toothbrush images.

- **Model Type**:  
  Convolutional Neural Network (CNN) — ideal for image classification tasks.

---

## Usage

### Upload Image
1. Launch the Streamlit app.
2. Select the **Upload Image** option.
3. Upload an image of a toothbrush.
4. The app will display whether the toothbrush is **Normal** or **Defective**.

### Live Camera Feed (Bonus)
1. Launch the Streamlit app.
2. Check the **Start Camera** option to enable your webcam.
3. Show the toothbrush to the camera for real-time anomaly detection.

---

## Project Structure

```bash
├── app.py                    # Streamlit app file
├── model/
│   └── keras_model.h5         # Trained model exported from Teachable Machine
├── README.md                  # Project documentation
├── requirements.txt           # List of required packages (optional)
```
