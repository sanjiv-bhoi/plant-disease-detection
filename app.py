import os
import json

import numpy as np
import streamlit as st
import tensorflow as tf

from utils import pretty, decode, prepare, get_parts, gradcam, overlay

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "plant_model_lean.keras")
CLASSES_PATH = os.path.join(HERE, "class_names.json")
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")


@st.cache_resource(show_spinner="Loading model (the first load can take a minute)...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names, get_parts(model)


st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a photo of a single plant leaf. A fine-tuned MobileNetV2 model "
    "classifies it into one of **38** healthy / diseased categories across 14 crops."
)

with st.sidebar:
    st.header("About")
    st.write(
        "- Model: MobileNetV2 (transfer learning + fine-tuning)\n"
        "- Data: New Plant Diseases Dataset (PlantVillage-based)\n"
        "- Validation accuracy: **97.84%** (17,572 images)"
    )
    st.header("Limitations")
    st.write(
        "- Trained on single leaves photographed against plain backgrounds; "
        "field photos may perform worse.\n"
        "- It always picks one of the 38 known classes, even for unrelated images.\n"
        "- Some tomato and corn diseases look alike and are sometimes confused.\n"
        "- This is a student project, not a substitute for expert diagnosis."
    )

model, class_names, parts = load_assets()

file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if file is not None:
    try:
        image = decode(file.getvalue())
    except Exception:
        st.error("Could not read that file as an image. Please upload a JPG or PNG.")
        st.stop()

    batch = prepare(image)
    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)[::-1][:3]
    top = int(order[0])
    confidence = float(probs[top])

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image")
    with col2:
        st.subheader("Prediction")
        st.success(pretty(class_names[top]))
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        if confidence < LOW_CONFIDENCE:
            st.warning("Low confidence. Try a clearer, well-lit photo of a single leaf.")

    st.write("**Top 3 predictions**")
    for i in order:
        st.write(pretty(class_names[int(i)]))
        st.progress(float(probs[int(i)]))

    if st.checkbox("Show Grad-CAM heatmap (where the model looked)"):
        cam = gradcam(batch, parts)
        st.image(
            overlay(batch, cam),
            caption="Red/yellow = regions that most increased the score of the predicted class",
        )
else:
    st.info("Upload a JPG or PNG leaf image to get a prediction.")
