import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model_high_acc.keras")

st.write(f"Model path: {MODEL_PATH}")
st.write(f"File exists: {os.path.exists(MODEL_PATH)}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")
    st.stop()

disease_names = [
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

st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="wide")
st.title("🌿 Plant Disease Classifier (Single Prediction)")

uploaded_file = st.file_uploader("📷 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Force image into RGB (3 channels)
    image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    # Guarantee 4D input with 3 channels
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array]*3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    top_idx = np.argmax(prediction)
    predicted_disease = disease_names[top_idx]
    confidence = float(prediction[0][top_idx]) * 100

    st.subheader("✅ Prediction Result")
    st.markdown(f"**Predicted Disease:** 🌱 `{predicted_disease}`")
    st.markdown(f"**Confidence (Accuracy):** `{confidence:.2f}%`")
    st.progress(min(int(confidence), 100))
else:
    st.info("👆 Please upload a clear leaf image to analyze.")
