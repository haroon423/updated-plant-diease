import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="🌿 Plant Disease Detection System", layout="centered")

st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image to identify the disease type and confidence.")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        st.write("Running load_model().")
        st.write(f"📂 Current directory: {os.getcwd()}")
        st.write(f"📁 Available files: {os.listdir(os.getcwd())}")

        model_path = "plant_disease_model_high_acc.keras"
        if not os.path.exists(model_path):
            st.error(f"❌ File not found: {model_path}")
            return None

        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        st.write(f"🔹 Model Input Shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None


model = load_model()

# ---------------------------
# Class Names (89 total)
# ---------------------------
class_names = [
    "apple black rot","apple leaf","apple mosaic virus","apple rust","apple scab",
    "banana leaf","banana panama disease","basil downy mildew","basil leaf",
    "bean halo blight","bean leaf","bean mosaic virus","bean rust","bell pepper leaf",
    "bell pepper leaf spot","blueberry leaf","blueberry rust","broccoli downy mildew",
    "broccoli leaf","cabbage alternaria leaf spot","cabbage leaf","carrot cavity spot",
    "cauliflower alternaria leaf spot","cauliflower leaf","celery anthracnose",
    "celery early blight","celery leaf","cherry leaf","cherry leaf spot",
    "cherry powdery mildew","citrus canker","citrus greening disease","coffee leaf",
    "coffee leaf rust","corn gray leaf spot","corn leaf","corn northern leaf blight",
    "corn rust","corn smut","cucumber angular leaf spot","cucumber bacterial wilt",
    "cucumber leaf","cucumber powdery mildew","eggplant cercospora leaf spot",
    "eggplant leaf","garlic leaf","garlic leaf blight","garlic rust","ginger leaf",
    "ginger leaf spot","ginger sheath blight","grape black rot","grape downy mildew",
    "grape leaf","grape leaf spot","grapevine leafroll disease","lettuce downy mildew",
    "lettuce leaf","lettuce mosaic virus","maple leaf","maple tar spot","peach leaf",
    "peach leaf curl","plum leaf","plum pocket disease","potato early blight",
    "potato late blight","potato leaf","raspberry leaf","rice blast","rice leaf",
    "rice sheath blight","soybean leaf","squash leaf","squash powdery mildew",
    "strawberry anthracnose","strawberry leaf","strawberry leaf scorch","tobacco leaf",
    "tobacco mosaic virus","tomato bacterial leaf spot","tomato early blight",
    "tomato late blight","tomato leaf","tomato leaf mold","tomato mosaic virus",
    "tomato septoria leaf spot","tomato yellow leaf curl virus","zucchini yellow mosaic virus"
]

# ---------------------------
# Prediction Function
# ---------------------------
def predict_disease(image):
    if model is None:
        st.warning("⚠️ Model not loaded. Cannot predict.")
        return None, None

    img = image.convert("RGB")  # ✅ Force 3 channels
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_idx, confidence


# ---------------------------
# Streamlit UI
# ---------------------------
if model:
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Analyzing disease..."):
            class_idx, confidence = predict_disease(image)
            if class_idx is not None:
                st.subheader(f"🩺 Predicted Disease: {class_names[class_idx].title()}")
                st.write(f"🎯 Confidence: {confidence*100:.2f}%")
            else:
                st.error("❌ Could not predict disease. Please try again.")
else:
    st.warning("⚠️ Model not loaded. Please ensure your `.keras` model file is in the same directory.")
