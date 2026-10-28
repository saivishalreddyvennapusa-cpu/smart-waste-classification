
import os
from pathlib import Path

# Windows GPU DLL setup must happen before importing TensorFlow.
_DLL_HANDLE = None
if os.name == "nt" and os.environ.get("CONDA_PREFIX"):
    dll_path = Path(os.environ["CONDA_PREFIX"]) / "Library" / "bin"
    if dll_path.exists():
        _DLL_HANDLE = os.add_dll_directory(str(dll_path))

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# ---------------- CONFIGURATION ----------------
ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = (
    ROOT.parent / "models" / "waste_efficientnetb0_v5_finetuned.weights.h5"
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
        "Keep separate from household rubbish. Use an authorised "
        "electronic-waste collection point."
    ),
    "automobile wastes": (
        "Keep vehicle-related waste separate. Contact an authorised "
        "automotive recycler or service centre."
    ),
    "battery waste": (
        "Do not put batteries in general waste. Use a designated battery "
        "collection point and avoid short-circuiting the terminals."
    ),
    "glass waste": (
        "Separate clean glass where locally accepted. Wrap broken glass "
        "securely and follow local collection rules."
    ),
    "light bulbs": (
        "Handle bulbs carefully. CFL and fluorescent bulbs may contain "
        "hazardous materials; use a designated collection point."
    ),
    "metal waste": (
        "Separate recyclable metal and take it to a local recycling "
        "collection point or authorised scrap dealer."
    ),
    "organic waste": (
        "Food and garden waste may be suitable for composting or municipal "
        "organic-waste collection."
    ),
    "paper waste": (
        "Keep paper clean and dry for recycling. Food-soiled paper may "
        "not be accepted."
    ),
    "plastic waste": (
        "Separate plastic and follow local recycling rules. Not every "
        "plastic type is recyclable."
    ),
}

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"V5 checkpoint not found: {WEIGHTS_PATH}"
        )

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=IMG_SIZE + (3,),
    )

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(
        len(CLASS_NAMES), activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)

    # Build model variables before restoring the trained weights.
    model(
        tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3)),
        training=False,
    )

    try:
        model.load_weights(str(WEIGHTS_PATH))
    except Exception as error:
        raise RuntimeError(
            "Could not load the trained V5 weights. "
            f"TensorFlow version: {tf.__version__}. "
            "The checkpoint format may be incompatible with "
            "the installed Keras version. Original error: "
            f"{error}"
        ) from error

    return model

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(pil_image):
    image = pil_image.convert("RGB")
    image = image.resize(IMG_SIZE)

    # Keep pixels in 0-255 range, matching the training pipeline.
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
st.caption("EfficientNetB0 V5 | Nine waste categories | Local inference")

st.info(
    "The model achieved 86.40% test accuracy on your recorded test set. "
    "Predictions may be incorrect; use them as guidance, not as a safety guarantee."
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

        # Compatible with Streamlit 1.22.0.
        st.image(
            display_image,
            caption="Image used for prediction",
            use_column_width=True,
        )

        with st.spinner("Loading model and classifying image..."):
            model = load_model()
            probabilities = model.predict(batch, verbose=0)[0]

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
                "Confidence is relatively low. Try a clearer image with "
                "one main waste item and better lighting."
            )

        st.subheader("Top 3 predictions")

        top_indices = np.argsort(probabilities)[::-1][:3]

        for index in top_indices:
            category = CLASS_NAMES[int(index)]
            score = float(probabilities[index])

            st.write(f"**{category}** — {score * 100:.2f}%")
            st.progress(score)

        st.subheader("Suggested handling")
        st.write(DISPOSAL_GUIDANCE[best_class])

        st.caption(
            "Always follow local waste-management rules, especially for "
            "batteries, electronic waste and light bulbs."
        )

    except Exception as error:
        st.error(f"Could not classify this image: {error}")

else:
    st.write(
        "Upload an image or select the camera option to classify a waste item."
    )

st.divider()
st.caption("Project: Deep Learning-Based Waste Classification Using Computer Vision")
