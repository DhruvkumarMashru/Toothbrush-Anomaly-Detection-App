from PIL import Image  # Correct import for handling images
import numpy as np
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Function to load the trained model
def load_trained_model(model_path):
    try:
        model = load_model(model_path)  # Load the model using Keras
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Preprocess image function for Streamlit image upload or file path
def preprocess_image(img_file):
    try:
        if hasattr(img_file, "read"):  # If the input is an uploaded file (from Streamlit)
            img = Image.open(io.BytesIO(img_file.read()))  # Open image from the uploaded file (in memory)
        else:
            img = image.load_img(img_file, target_size=(224, 224))  # If it's a file path, load and resize
        
        img = img.resize((224, 224))  # Ensure the image is of the correct size (optional if already done)
        img_array = image.img_to_array(img)  # Convert the image to a numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Expand the dimensions to match model input
        img_array /= 255.0  # Normalize the image array to scale pixel values to [0, 1]
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Example code to use the model for prediction
def predict_image(image_input, model):
    try:
        img_array = preprocess_image(image_input)  # Preprocess the image
        if img_array is not None:
            prediction = model.predict(img_array)  # Perform prediction
            predicted_class = np.argmax(prediction)  # Get the predicted class (index of highest probability)
            confidence = np.max(prediction)  # Get the confidence score (highest probability)
            return predicted_class, confidence
        else:
            print("Error: Image preprocessing failed.")
            return None, None
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None

# Example usage (for testing)
model_path = 'model/keras_model.h5'  # Example model path
model = load_trained_model(model_path)

if model:
    image_input = 'path_to_image.jpg'  # Replace with an actual image path or Streamlit uploaded file
    predicted_class, confidence = predict_image(image_input, model)
    if predicted_class is not None:
        print(f"Prediction: {predicted_class}, Confidence: {confidence}")
    else:
        print("Prediction failed.")
