import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model_high_acc.keras")

st.write("Model path:", MODEL_PATH)
st.write("File exists:", os.path.exists(MODEL_PATH))

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")

disease_names = [...]  # your 89 labels

st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="wide")
st.title("🌿 Plant Disease Classifier (Single Prediction)")

uploaded_file = st.file_uploader("📷 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((224, 224))  # update this later if model.input_shape differs
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        predictions = model.predict(img_array)[0]
        top_idx = np.argmax(predictions)
        predicted_disease = disease_names[top_idx]
        confidence = float(predictions[top_idx]) * 100

        st.subheader("✅ Prediction Result")
        st.markdown(f"**Predicted Disease:** 🌱 `{predicted_disease}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")
        st.progress(min(int(confidence), 100))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
