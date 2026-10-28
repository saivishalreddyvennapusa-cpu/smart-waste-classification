
import os
from pathlib import Path

# ---------------- WINDOWS TENSORFLOW DLL SETUP ----------------
_DLL_HANDLE = None

if os.name == "nt" and os.environ.get("CONDA_PREFIX"):
    dll_path = Path(os.environ["CONDA_PREFIX"]) / "Library" / "bin"
    if dll_path.exists():
        _DLL_HANDLE = os.add_dll_directory(str(dll_path))

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ---------------- CONFIGURATION ----------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT.parent / "models" / "waste_efficientnetb0_v5_fullmodel.h5"
)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "E-waste",
    "automobile wastes",
    "battery waste",
    "glass waste",
    "light bulbs",
    "metal waste",
    "organic waste",
    "paper waste",
    "plastic waste",
]

DISPOSAL_GUIDANCE = {
    "E-waste": (
        "Take electronic devices to an authorized e-waste collection "
        "centre. Do not put electronic components in regular household waste."
    ),
    "automobile wastes": (
        "Use an authorized automotive recycling or service centre. "
        "Do not dispose of automotive fluids or parts carelessly."
    ),
    "battery waste": (
        "Take batteries to an authorized battery collection or recycling "
        "point. Do not burn, puncture, or place them in regular waste."
    ),
    "glass waste": (
        "Separate glass for recycling where accepted. Handle broken glass "
        "carefully and package it safely."
    ),
    "light bulbs": (
        "Take fluorescent and other specialist bulbs to an appropriate "
        "collection point. Avoid breaking them."
    ),
    "metal waste": (
        "Separate clean metal for recycling and take it to an accepted "
        "scrap or recycling collection point."
    ),
    "organic waste": (
        "Place suitable food scraps and garden waste in a composting or "
        "municipal organic-waste collection system."
    ),
    "paper waste": (
        "Keep paper dry and clean, then place it in the paper-recycling "
        "stream where available."
    ),
    "plastic waste": (
        "Check local recycling rules, empty the container, and separate "
        "accepted plastics for collection."
    ),
}


# ---------------- LOAD CONVERTED MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Converted model not found: {MODEL_PATH}. "
            "Check that the model file is committed to GitHub."
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    return model


# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(pil_image):
    image = pil_image.convert("RGB")
    image = image.resize(IMG_SIZE)

    # Match the original training pipeline: pixel values remain 0–255.
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    return batch, image


# ---------------- USER INTERFACE ----------------
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ Smart Waste Classification")

st.caption(
    "EfficientNetB0 V5 | Nine waste categories | Image classification"
)

st.info(
    "The model achieved 86.40% test accuracy on the recorded test set. "
    "Predictions can be incorrect and should be used as guidance, "
    "not as a safety guarantee."
)

image_source = st.radio(
    "Choose image source",
    ["Upload image", "Take photo with camera"],
    index=0,
)

image_file = None

if image_source == "Upload image":
    image_file = st.file_uploader(
        "Choose a waste image",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    image_file = st.camera_input("Capture a waste image")


if image_file is not None:
    try:
        uploaded_image = Image.open(image_file)
        batch, display_image = preprocess_image(uploaded_image)

        st.image(
            display_image,
            caption="Image used for prediction",
            use_container_width=True,
        )

        with st.spinner("Loading model and classifying image..."):
            model = load_model()
            probabilities = model.predict(batch, verbose=0)[0]

        if len(probabilities) != len(CLASS_NAMES):
            raise ValueError(
                f"Expected {len(CLASS_NAMES)} class scores, "
                f"but received {len(probabilities)}."
            )

        best_index = int(np.argmax(probabilities))
        best_class = CLASS_NAMES[best_index]
        confidence = float(probabilities[best_index])

        st.subheader("Prediction")
        st.success("Predicted category: " + best_class)

        st.metric(
            label="Model confidence",
            value=f"{confidence * 100:.2f}%",
        )

        if confidence < 0.60:
            st.warning(
                "Confidence is relatively low. Try a clearer image "
                "with one main waste item and better lighting."
            )

        st.subheader("Top 3 predictions")

        top_indices = np.argsort(probabilities)[::-1][:3]

        for index in top_indices:
            category = CLASS_NAMES[int(index)]
            score = float(probabilities[index])

            st.write(f"**{category}** — {score * 100:.2f}%")
            st.progress(float(np.clip(score, 0.0, 1.0)))

        st.subheader("Suggested handling")
        st.write(DISPOSAL_GUIDANCE[best_class])

        st.caption(
            "Always follow local waste-management rules, especially "
            "for batteries, electronic waste, and light bulbs."
        )

    except Exception as error:
        st.error(f"Could not classify this image: {error}")

else:
    st.write(
        "Upload an image or select the camera option to classify a waste item."
    )

st.divider()

st.caption(
    "Project: Deep Learning-Based Waste Classification Using Computer Vision"
)
