import streamlit as st
from PIL import Image
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from ela import perform_ela  # Make sure this function exists in ela.py

# Load the trained model
model = joblib.load("model.pkl")  # Change this to your model's path

# Streamlit UI
st.title("Digital Forgery Detection")
st.write("Upload an image")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform ELA
    ela_image = perform_ela(image)
    ela_array = np.array(ela_image.convert("L"))  # Convert to grayscale

    # Show ELA heatmap
    st.write("ELA Deviation Heatmap:")
    fig, ax = plt.subplots()
    sns.heatmap(ela_array, cmap='viridis', xticklabels=False, yticklabels=False, cbar=False, ax=ax)
    st.pyplot(fig)

    # Resize and preprocess the image for model
    image = image.resize((256, 256))  # Resize as per the model input size
    image_array = np.array(image).astype('float32') / 255.0
    features = image_array.flatten().reshape(1, -1)

    # Get the prediction and confidence
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max()  # Get the highest confidence score

    # Display the results
    st.write(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
