import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ----------------------------
# 1. Load .keras Model
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model_high_acc.keras")

st.write("Model path:", os.path.abspath(MODEL_PATH))
st.write("File exists:", os.path.exists(MODEL_PATH))

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")
    st.stop()

# ----------------------------
# 2. Disease Names (89)
# ----------------------------
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

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="wide")
st.title("🌿 Plant Disease Classifier (Using .keras Model)")

uploaded_file = st.file_uploader("📷 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----------------------------
    # 4. Preprocess the image
    # ----------------------------
    img_resized = image.resize((225, 225))  # Must match model input
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 225, 225, 3)

    # ----------------------------
    # 5. Predict
    # ----------------------------
    predictions = model.predict(img_array)
    top_idx = np.argmax(predictions[0])
    predicted_disease = disease_names[top_idx]
    confidence = float(predictions[0][top_idx]) * 100

    # ----------------------------
    # 6. Display Results
    # ----------------------------
    st.subheader("✅ Prediction Result")
    st.markdown(f"**Predicted Disease:** 🌱 `{predicted_disease}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")
    st.progress(min(int(confidence), 100))
else:
    st.info("👆 Please upload a clear leaf image to analyze.")
