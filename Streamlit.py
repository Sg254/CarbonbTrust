# app_streamlit.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.title("MNIST digit classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn.h5")

model = load_model()

st.write("Draw a digit (0-9) or upload an image.")
canvas_file = st.file_uploader("Upload a 28x28 grayscale or color image (.png/.jpg)")

if canvas_file is not None:
    img = Image.open(canvas_file).convert("L")
    img = ImageOps.invert(img)  # if white background + black digit, invert if needed
    img = img.resize((28,28))
    st.image(img, caption="Input (28x28 grayscale)", width=100)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1,28,28,1)
    probs = model.predict(arr)[0]
    pred = int(np.argmax(probs))
    st.write(f"Prediction: **{pred}** (confidence {probs[pred]:.2f})")
    st.bar_chart(probs)
else:
    st.write("Upload a single image file to classify.")
  
