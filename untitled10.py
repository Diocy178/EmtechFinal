import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Function to load the Keras model
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to predict the class of an image
def predict_image(img_path, model, class_labels, threshold=0.5):
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]
        
        if confidence < threshold:
            return "Unknown Weather", confidence
        else:
            return class_labels[class_index], confidence
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None, None

# Streamlit app
st.title("Weather Image Classification")

# Load the model
model_path = 'finals_model (1).h5'
model = load_keras_model(model_path)

if model is not None:
    # Define the class labels
    class_labels = ['Rain', 'Shine', 'Cloudy', 'Sunrise', 'Unknown']

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image
        img_path = os.path.join("uploaded_image.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(img_path, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Predict the class of the image
        label, confidence = predict_image(img_path, model, class_labels, threshold=0.8)
        
        if label is not None and confidence is not None:
            # Display the prediction
            st.write(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}")

# Add gradient background related to weather
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #72edf2 10%, #5151e5 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)
