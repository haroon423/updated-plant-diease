import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("🌿 Plant Disease Detection System (89 Classes)")

@st.cache_resource
def load_keras_model(model_path="plant_disease_model_high_acc.keras"):
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        st.info(f"Model Input Shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_keras_model()

# ✅ Full 89 class names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two_spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
    'Apple___Alternaria_leaf_spot', 'Apple___Anthracnose', 'Apple___Frogeye_leaf_spot',
    'Banana___Black_Sigatoka', 'Banana___Yellow_Sigatoka', 'Banana___Panama_disease', 
    'Banana___Moko_disease', 'Banana___healthy', 'Citrus___Canker', 'Citrus___Greening', 
    'Citrus___Melanose', 'Citrus___Sooty_mold', 'Citrus___healthy', 'Coffee___Rust', 
    'Coffee___Leaf_Miner', 'Coffee___healthy', 'Cotton___Bacterial_blight', 'Cotton___Fusarium_wilt', 
    'Cotton___healthy', 'Paddy___Brown_spot', 'Paddy___Leaf_blast', 'Paddy___Leaf_scald', 
    'Paddy___Sheath_blight', 'Paddy___healthy', 'Sugarcane___Red_rot', 'Sugarcane___Smut', 
    'Sugarcane___healthy', 'Wheat___Brown_rust', 'Wheat___Yellow_rust', 'Wheat___Leaf_blight', 
    'Wheat___Septoria_leaf_spot', 'Wheat___healthy', 'Maize___Downy_mildew', 
    'Maize___Curvularia_leaf_spot', 'Maize___healthy', 'Tea___Brown_blight', 
    'Tea___Gray_blight', 'Tea___Red_leaf_spot', 'Tea___healthy', 'Tomato___Fusarium_wilt', 
    'Tomato___Verticillium_wilt', 'Pepper,_bell___Anthracnose', 'Papaya___Ring_spot', 
    'Papaya___Anthracnose', 'Papaya___Powdery_mildew', 'Papaya___healthy', 
    'Mango___Anthracnose', 'Mango___Bacterial_Canker', 'Mango___Sooty_mold', 'Mango___healthy'
]

st.info(f"✅ Total Classes Loaded: {len(CLASS_NAMES)}")

def predict_disease(image):
    if model is None:
        st.error("⚠️ Model not loaded.")
        return None, None

    img = image.resize((224, 224))
    img_array = np.array(img)

    # ✅ Convert grayscale → RGB
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    with st.spinner("🔍 Analyzing..."):
        predicted_class, confidence = predict_disease(image)
    if predicted_class:
        st.success(f"🌿 **Prediction:** {predicted_class}")
        st.info(f"📊 Confidence: {confidence:.2f}%")
