import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image to identify the disease type and confidence.")

@st.cache_resource
def load_model():
    st.write("Running load_model().")
    cwd = os.getcwd()
    st.write(f"\n📂 Current directory: {cwd}\n")
    st.write(f"\n📁 Available files: {os.listdir(cwd)}\n")

    model_path = "plant_disease_model_high_acc.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ File not found: {model_path}")
        st.warning("⚠️ Model not loaded. Please ensure your .keras model file is in the same directory.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Open and ensure RGB (3 channels)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Resize to 224x224 — standard input size for most CNNs (EfficientNet, ResNet, etc.)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Optional: Debug shape (uncomment if needed)
    # st.write(f"Input tensor shape: {img_array.shape}")

    # Make prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)

    st.success(f"🌿 Predicted Disease: **{class_labels[class_idx]}**")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")

elif uploaded_file is not None and model is None:
    st.warning("⚠️ Model not loaded. Please check your .keras file.")
