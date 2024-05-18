import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil

def predict_image(img_path, model, class_labels, threshold=0.6):
    try:
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        class_label = class_labels[class_index]
        confidence = predictions[0][class_index]
        if confidence < threshold:
            class_label = "Unknown"
        return class_label, confidence
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None, None

def main():
    st.title("Image Classification App")

    model_path = 'finals_model (1).h5'
    if not os.path.exists(model_path):
        st.error(f"No file or directory found at {model_path}")
        return

    model = load_model(model_path)
    class_labels = ['Rain', 'Sunrise', 'Cloudy', 'Shine']

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        img_path = os.path.join("temp", uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        label, confidence = predict_image(img_path, model, class_labels)

        if label is not None and confidence is not None:
            st.write(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}")

            # Save the image to the corresponding folder
            if label != "Unknown":
                destination_folder = os.path.join("data", label)
                os.makedirs(destination_folder, exist_ok=True)
                shutil.move(img_path, os.path.join(destination_folder, os.path.basename(img_path)))
                st.write(f"Image moved to {label} folder.")
            else:
                unknown_folder = os.path.join("data", "Unknown")
                os.makedirs(unknown_folder, exist_ok=True)
                shutil.move(img_path, os.path.join(unknown_folder, os.path.basename(img_path)))
                st.write("Image classified as Unknown and moved to Unknown folder.")

            # Display the current working directory
            current_directory = os.getcwd()
            st.write(f"Current Directory: {current_directory}")

            # Display the directory where the images are moved
            moved_directory = os.path.abspath(os.path.join(current_directory, "data"))
            st.write(f"Images moved to: {moved_directory}")

if __name__ == "__main__":
    # Create temp directory for uploads
    if not os.path.exists("temp"):
        os.makedirs("temp")

    main()
