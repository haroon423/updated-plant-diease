import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

# ----------------------------
# 1. Load TFLite model
# ----------------------------
MODEL_PATH = "model_compressed.tflite"  # replace with your TFLite model path
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# 2. Exact 89 disease names
# ----------------------------
disease_names = [
    "apple black rot",
    "apple leaf",
    "apple mosaic virus",
    "apple rust",
    "apple scab",
    "banana leaf",
    "banana panama disease",
    "basil downy mildew",
    "basil leaf",
    "bean halo blight",
    "bean leaf",
    "bean mosaic virus",
    "bean rust",
    "bell pepper leaf",
    "bell pepper leaf spot",
    "blueberry leaf",
    "blueberry rust",
    "broccoli downy mildew",
    "broccoli leaf",
    "cabbage alternaria leaf spot",
    "cabbage leaf",
    "carrot cavity spot",
    "cauliflower alternaria leaf spot",
    "cauliflower leaf",
    "celery anthracnose",
    "celery early blight",
    "celery leaf",
    "cherry leaf",
    "cherry leaf spot",
    "cherry powdery mildew",
    "citrus canker",
    "citrus greening disease",
    "coffee leaf",
    "coffee leaf rust",
    "corn gray leaf spot",
    "corn leaf",
    "corn northern leaf blight",
    "corn rust",
    "corn smut",
    "cucumber angular leaf spot",
    "cucumber bacterial wilt",
    "cucumber leaf",
    "cucumber powdery mildew",
    "eggplant cercospora leaf spot",
    "eggplant leaf",
    "garlic leaf",
    "garlic leaf blight",
    "garlic rust",
    "ginger leaf",
    "ginger leaf spot",
    "ginger sheath blight",
    "grape black rot",
    "grape downy mildew",
    "grape leaf",
    "grape leaf spot",
    "grapevine leafroll disease",
    "lettuce downy mildew",
    "lettuce leaf",
    "lettuce mosaic virus",
    "maple leaf",
    "maple tar spot",
    "peach leaf",
    "peach leaf curl",
    "plum leaf",
    "plum pocket disease",
    "potato early blight",
    "potato late blight",
    "potato leaf",
    "raspberry leaf",
    "rice blast",
    "rice leaf",
    "rice sheath blight",
    "soybean leaf",
    "squash leaf",
    "squash powdery mildew",
    "strawberry anthracnose",
    "strawberry leaf",
    "strawberry leaf scorch",
    "tobacco leaf",
    "tobacco mosaic virus",
    "tomato bacterial leaf spot",
    "tomato early blight",
    "tomato late blight",
    "tomato leaf",
    "tomato leaf mold",
    "tomato mosaic virus",
    "tomato septoria leaf spot",
    "tomato yellow leaf curl virus",
    "zucchini yellow mosaic virus"
]

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.title("Plant Disease Classifier (89 Classes)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(input_details[0]['dtype']))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Predicted class
    top_idx = np.argmax(output_data)
    predicted_disease = disease_names[top_idx]
    confidence = output_data[top_idx] * 100

    st.subheader("Predicted Disease")
    st.write(f"**{predicted_disease}** with confidence: **{confidence:.2f}%**")

    # Top 5 predictions
    st.subheader("Top 5 Predictions")
    top_5_idx = np.argsort(output_data)[::-1][:5]
    top5_df = pd.DataFrame({
        "Disease": [disease_names[i] for i in top_5_idx],
        "Confidence (%)": [output_data[i]*100 for i in top_5_idx]
    })
    st.dataframe(top5_df)

    # Optional: Full 89-class table
    if st.checkbox("Show all 89 classes"):
        df_all = pd.DataFrame({
            "Disease": disease_names,
            "Confidence (%)": output_data*100
        }).sort_values(by="Confidence (%)", ascending=False)
        st.dataframe(df_all)