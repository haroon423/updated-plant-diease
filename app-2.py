import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="🌿 Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection System")

@st.cache_resource
def load_model():
    model_path = "plant_disease_model_high_acc.keras"
    st.write("📂 Current directory:", os.getcwd())
    st.write("📁 Available files:", os.listdir())

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()

    try:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        st.success("✅ Model loaded successfully!")
        st.write("🔹 Model input shape:", model.input_shape)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

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

def predict(image):
    img = image.convert("RGB")  # ✅ Force 3 channels
    img = img.resize((224, 224))  # ✅ Match model input size
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return predicted_class, confidence

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        pred, conf = predict(img)

    st.success(f"✅ Prediction: **{pred}** ({conf:.2f}%)")
