from PIL import Image  # Correct import
import numpy as np
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_trained_model(model_path):
    model = load_model(model_path)
    return model


def preprocess_image(img_file):
    if hasattr(img_file, "read"):  # if it's an UploadedFile (like in Streamlit)
        img = Image.open(io.BytesIO(img_file.read()))  # Open image from uploaded file
    else:
        img = image.load_img(img_file, target_size=(224, 224))  # If it's a file path

    img = img.resize((224, 224))  # Ensure image size is correct (optional if using target_size in load_img)
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size of 1
    img_array /= 255.  # Normalize the image array
    return img_array
