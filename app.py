import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Load your trained model
# -------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_11_layer.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Image size used during training
IMG_SIZE = (224, 224)

# Class labels (update based on your dataset)
class_names = ["glioma", "meningioma", "notumor", "pituitary"]  
# Change order according to train_generator.class_indices

# -------------------------------
# Title
# -------------------------------
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image and the model will predict the tumor type.")

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    # Ensure correct shape: (1, 224, 224, 3)
    if img.ndim == 2:
        img = np.stack((img,)*3, axis=-1)

    if img.shape[-1] == 1:
        img = np.concatenate([img]*3, axis=-1)

    img = np.expand_dims(img, axis=0)

    # Predict button
    if st.button("🔍 Predict Tumor Type"):
        with st.spinner("Analyzing the image..."):
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

        st.success(f"### 🧾 Prediction: **{class_names[predicted_class].upper()}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")

        # Show probability distribution
        st.subheader("Class Probabilities")
        for idx, prob in enumerate(predictions[0]):
            st.write(f"{class_names[idx]}: **{prob*100:.2f}%**")