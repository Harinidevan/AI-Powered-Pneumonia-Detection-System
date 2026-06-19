import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
</style>
""", unsafe_allow_html=True)

model = load_model("densenet_pneumonia.h5")

st.markdown("""
<h1 style='text-align:center;
color:#1f77b4;
font-size:65px;
font-weight:900;'>
🩺 AI-Powered Pneumonia Detection System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='text-align:center;
color:#555;'>
DenseNet121-Based Chest X-Ray Analysis
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background: linear-gradient(90deg,#1f77b4,#4facfe);
padding:15px;
border-radius:10px;
text-align:center;
color:white;
font-size:20px;
font-weight:bold;">
AI-Powered Medical Imaging for Pneumonia Detection
</div>
""", unsafe_allow_html=True)

st.write("")

st.sidebar.title("📊 Model Performance")

st.sidebar.success("DenseNet121")

st.sidebar.metric("Accuracy", "95.80%")
st.sidebar.metric("Precision", "94.55%")
st.sidebar.metric("Recall", "97.20%")
st.sidebar.metric("AUC", "99.07%")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Harini")

col1, col2 = st.columns([1, 1])

with col1:

    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-Ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

with col2:

    if uploaded_file:

        img = Image.open(uploaded_file)
        img = img.resize((224, 224))

        img_array = np.array(img)

        # Convert grayscale to RGB
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)

        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        st.markdown("## 🔍 Prediction Result")

        if prediction > 0.5:

            confidence = prediction * 100

            st.error("⚠️ Pneumonia Detected")

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(confidence) / 100)

        else:

            confidence = (1 - prediction) * 100

            st.success("✅ Normal Chest X-Ray")

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(confidence) / 100)

st.markdown("---")

st.markdown("""
## 📌 About This Project

This AI system analyzes chest X-ray images using a DenseNet121 deep learning model.

### Features
✅ Chest X-Ray Classification

✅ Pneumonia Detection

✅ Confidence Score Prediction

✅ Streamlit Web Interface

✅ Model Accuracy: 95.8%

""")

st.markdown("---")

st.markdown("""
<div style='text-align:center; color:gray;'>
Developed by Harini | AI & Data Science Project |
DenseNet121 + TensorFlow + Streamlit
</div>
""", unsafe_allow_html=True)