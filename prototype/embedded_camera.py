"""
USB-camera prediction demo for a Raspberry Pi / Linux computer.
This uses the same V5 weights-only checkpoint and model architecture.
Run from the prototype folder after installing compatible TensorFlow, OpenCV,
NumPy and Pillow for the target device.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "E-waste", "automobile wastes", "battery waste", "glass waste",
    "light bulbs", "metal waste", "organic waste", "paper waste", "plastic waste"
]
ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT.parent / "models" / "waste_efficientnetb0_v5_finetuned.weights.h5"

def build_model():
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.10),
    ], name="augmentation")
    base = EfficientNetB0(include_top=False, weights=None, input_shape=IMG_SIZE + (3,))
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    inp = layers.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = models.Model(inp, out)
    model(tf.zeros((1, 224, 224, 3)), training=False)
    model.load_weights(str(WEIGHTS_PATH))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    args = parser.parse_args()

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Checkpoint missing: {WEIGHTS_PATH}")
    model = build_model()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check USB camera connection and camera index.")

    print("Camera ready. Press SPACE to classify current frame; press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a camera frame.")
            break
        preview = frame.copy()
        cv2.putText(preview, "SPACE: classify | Q: quit", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("Waste classifier camera", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, IMG_SIZE)
            batch = np.expand_dims(resized.astype(np.float32), axis=0)
            probs = model.predict(batch, verbose=0)[0]
            idx = int(np.argmax(probs))
            print(f"Prediction: {CLASS_NAMES[idx]} ({probs[idx] * 100:.2f}%)")
            text = f"{CLASS_NAMES[idx]}: {probs[idx] * 100:.1f}%"
            cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
            cv2.imshow("Waste classifier camera", frame)
            cv2.waitKey(1000)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
