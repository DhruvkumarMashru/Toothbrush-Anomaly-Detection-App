from PIL import Image  # Correct import for handling images
import numpy as np
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Function to load the trained model
def load_trained_model(model_path):
    model = load_model(model_path)  # Load the model using Keras
    return model

# Preprocess image function for Streamlit image upload or file path
def preprocess_image(img_file):
    if hasattr(img_file, "read"):  # If the input is an uploaded file (from Streamlit)
        img = Image.open(io.BytesIO(img_file.read()))  # Open image from the uploaded file (in memory)
    else:
        img = image.load_img(img_file, target_size=(224, 224))  # If it's a file path, load and resize

    img = img.resize((224, 224))  # Ensure the image is of the correct size (optional if already done)
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand the dimensions to match model input
    img_array /= 255.0  # Normalize the image array to scale pixel values to [0, 1]
    
    return img_array
