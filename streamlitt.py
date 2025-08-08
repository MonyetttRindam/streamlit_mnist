import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
@st.cache_resource  # agar tidak load model setiap kali refresh
def load_mnist_model():
    return load_model("mnist_model.h5")

model = load_mnist_model()

# Judul aplikasi
st.title("MNIST Digit Classifier")
st.markdown("Upload gambar angka tulisan tangan (digit 0–9), dan model akan memprediksi angkanya.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka dan tampilkan gambar
    image = Image.open(uploaded_file).convert("L")  # konversi ke grayscale
    st.image(image, caption="Gambar yang diupload", width=150)

    # Preprocessing
    image = image.resize((28, 28))  # sesuaikan ukuran
    img_array = np.array(image)
    img_array = img_array.reshape(1, 28, 28, 1).astype("float32") / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Tampilkan hasil
    st.subheader(f"Prediksi: {predicted_class}")
    st.write("Confidence per angka:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{i} : {prob:.4f}")

st.markdown(
    "Tugas Mini-Project Muhammad Abil Khoiri"
)
