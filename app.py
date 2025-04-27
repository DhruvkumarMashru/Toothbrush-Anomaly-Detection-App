import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from utils import preprocess_image  # Ensure preprocess_image function is defined correctly

# Load the model
def load_trained_model(model_path):
    model = load_model(model_path, compile=False)  # <-- Fixed here
    return model

# Load the model
model = load_trained_model('model/keras_model.h5')
class_names = ['Normal', 'Anomaly']

# Streamlit UI
st.title("🛡️ Anomaly Detector")

menu = ['Upload Image', 'Live Camera (Bonus)']
choice = st.sidebar.selectbox('Select Mode', menu)

# Function to handle image prediction
def predict_image(image_input):
    img_array = preprocess_image(image_input)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Adding messages based on class
    if predicted_class == 'Normal':
        message = "This is a good image."
    else:
        message = "This is a defective image."

    return predicted_class, confidence, message

# Upload Image Mode
if choice == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True, width=300)  # Adjust width here
        predicted_class, confidence, message = predict_image(uploaded_file)
        st.success(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
        st.write(message)

# Live Camera Mode
elif choice == 'Live Camera (Bonus)':
    st.warning("Allow camera access and click Start!")

    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    run = st.checkbox('Start Camera', value=st.session_state.camera_running)

    if run != st.session_state.camera_running:
        st.session_state.camera_running = run

    if run:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Failed to access camera.")
        else:
            st.info("Camera started. Capturing frames...")

            # Streamlit Image placeholder for the frame
            FRAME_WINDOW = st.image([])

            while run:  # This loop will keep updating the frame while the checkbox is checked
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to grab frame from camera.")
                    break

                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))

                # Use frame_resized directly for prediction
                img_array = np.expand_dims(frame_resized, axis=0) / 255.0
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Adding messages based on class
                if predicted_class == 'Normal':
                    message = "This is a good image."
                else:
                    message = "This is a defective image."

                # Draw label on the frame
                label = f"{predicted_class} ({confidence*100:.2f}%)"
                cv2.putText(frame_rgb, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the frame with prediction
                FRAME_WINDOW.image(frame_rgb)

                # Display the message
                st.write(message)

                # Stop capturing if checkbox is unchecked
                if not run:
                    break

            camera.release()

    else:
        st.write('Camera stopped')
