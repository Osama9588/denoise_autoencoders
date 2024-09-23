import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load your pre-trained model
model = load_model('denoising_model.h5')  # Replace with the actual path to your model

# Function to preprocess the image
def preprocess_image(image, target_size=(28, 28)):
    image = image.resize(target_size)  # Resize the image to match the input shape of the model
    image = image.convert('L')  # Convert image to grayscale (1 channel)
    image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (28, 28, 1)
    return np.expand_dims(image, axis=0)  # Add batch dimension (1, 28, 28, 1)

# Streamlit UI for image uploading and denoising
st.title("Image Denoising using Autoencoder")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Preprocess the image to match the input of the model
    processed_image = preprocess_image(image)

    # Use the model to denoise the image
    denoised_image = model.predict(processed_image)[0]  # Get the denoised image
    denoised_image = np.squeeze(denoised_image, axis=-1)  # Remove channel dimension

    # Post-process and display the denoised image
    denoised_image = (denoised_image * 255).astype(np.uint8)  # Scale back to [0, 255]
    denoised_image_pil = Image.fromarray(denoised_image)

    st.image(denoised_image_pil, caption="Denoised Image", use_column_width=True)
