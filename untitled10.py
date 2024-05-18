import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = 'finals_model.h5'
model = load_model(model_path)

# Define the class labels
class_labels = ['Rain', 'Sunrise', 'Cloudy', 'Shine']

# Function to predict the class of an image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    class_label = class_labels[class_index]
    confidence = predictions[0][class_index]
    return class_label, confidence

# Streamlit app
st.title("Weather Image Classification")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save the uploaded image
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(img_path, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Predict the class of the image
    label, confidence = predict_image(img_path)
    
    # Display the prediction
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")
