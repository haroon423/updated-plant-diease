import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 🌱 Load trained model
MODEL_PATH = "plant_disease_model_high_acc.keras"
st.write(f"Model path: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")
    st.stop()

# 🌾 Class labels (89 total)
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

# 🌿 Page config
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="wide")
st.title("🌱 Plant Disease Detection System")

# 📤 Upload Image
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Ensure image has 3 channels (RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"✅ **Prediction:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")
