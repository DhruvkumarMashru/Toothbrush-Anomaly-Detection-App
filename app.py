import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load the trained model
def load_trained_model(model_path):
    model = load_model(model_path, compile=False)  # Avoid re-compiling the model
    return model

# Preprocess image function
def preprocess_image(img_file):
    if hasattr(img_file, "read"):  # Check if it’s an uploaded file
        img = Image.open(io.BytesIO(img_file.read()))
    else:  # If it’s a file path
        img = image.load_img(img_file, target_size=(224, 224))

    img = img.resize((224, 224))  # Resize image to the expected model input size
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image array
    return img_array

# Prediction function
def predict_image(image_input):
    img_array = preprocess_image(image_input)  # Preprocess the uploaded image
    prediction = model.predict(img_array)  # Perform prediction

    # Assuming model output is a classification, using np.argmax to get the highest probability class
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Return predictions and confidence
    return predicted_class, confidence

# Load the model
model = load_trained_model('model/keras_model.h5')
class_names = ['Normal', 'Anomaly']

# Streamlit UI
st.title("🛡️ Anomaly Detector")

menu = ['Upload Image', 'Live Camera']
choice = st.sidebar.selectbox('Select Mode', menu)

# Upload Image Mode
if choice == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        predicted_class, confidence = predict_image(uploaded_file)
        st.success(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")

        # Adding message based on class
        if predicted_class == 'Normal':
            message = "This is a good image."
        else:
            message = "This is a defective image."
        
        st.write(message)

# Live Camera Mode using st.camera_input()
elif choice == 'Live Camera':
    st.warning("Allow camera access and click Start!")

    # Streamlit's camera input
    camera_input = st.camera_input("Capture Image")

    if camera_input is not None:
        # If an image is captured, use the image for prediction
        st.image(camera_input, caption='Captured Image', use_container_width=True)
        predicted_class, confidence = predict_image(camera_input)
        st.success(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")

        # Adding message based on class
        if predicted_class == 'Normal':
            message = "This is a good image."
        else:
            message = "This is a defective image."
        
        st.write(message)
