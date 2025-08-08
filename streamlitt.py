import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load model yang sudah dilatih
model = load_model("mnist_model.h5")

# Judul
st.title("MNIST Digit Classifier")
st.write("Upload gambar angka tangan (digit 0-9), dan model akan memprediksi angkanya.")

# Upload file gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing: resize dan normalisasi
    image = image.resize((28, 28))  # ukuran input MNIST
    img_array = np.array(image)
    img_array = img_array.reshape(1, 28, 28, 1)  # reshape ke batch x height x width x channel
    img_array = img_array.astype("float32") / 255.0  # normalisasi

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Output
    st.write(f"Prediksi: **{predicted_class[0]}**")
