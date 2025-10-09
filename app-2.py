import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ----------------------------
# 1. Load TFLite model
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_compressed.tflite")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# 2. Disease names (89)
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
st.title("🌿 Plant Disease Classifier (Single Prediction)")

uploaded_file = st.file_uploader("📷 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----------------------------
    # 4. Preprocess and predict
    # ----------------------------
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(input_details[0]['dtype']))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # ----------------------------
    # 5. Get top prediction
    # ----------------------------
    top_idx = np.argmax(output_data)
    predicted_disease = disease_names[top_idx]
    confidence = float(output_data[top_idx]) * 100

    # ----------------------------
    # 6. Display results
    # ----------------------------
    st.subheader("✅ Prediction Result")
    st.markdown(f"**Predicted Disease:** 🌱 `{predicted_disease}`")
    st.markdown(f"**Confidence (Accuracy):** `{confidence:.2f}%`")

    # Optional visual confidence bar
    st.progress(min(int(confidence), 100))

else:
    st.info("👆 Please upload a clear leaf image to analyze.")
