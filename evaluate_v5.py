
import os
from pathlib import Path

# Set up Windows TensorFlow GPU DLLs before importing TensorFlow.
if os.environ.get("CONDA_PREFIX"):
    _dll_handle = os.add_dll_directory(
        os.path.join(os.environ["CONDA_PREFIX"], "Library", "bin")
    )

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

TEST_DIR = Path("dataset/wastes/test")
WEIGHTS_PATH = Path(
    "models/waste_efficientnetb0_v5_finetuned.weights.h5"
)
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR.resolve()}")

if not WEIGHTS_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS_PATH.resolve()}")

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# Load test images without shuffling.
test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = test_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Rebuild exactly the same architecture used during training.
augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.10),
    ],
    name="augmentation"
)

base_model = EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=IMG_SIZE + (3,)
)
base_model.trainable = True

# Match the fine-tuning setup: only final 30 backbone layers trainable,
# and BatchNormalization layers remain frozen.
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
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Build model variables before loading the checkpoint.
model(tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3)), training=False)
model.load_weights(str(WEIGHTS_PATH))
print("Fine-tuned checkpoint loaded successfully.")

# Evaluate independent test data.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    digits=4,
    zero_division=0
)
print("\nCLASSIFICATION REPORT\n")
print(report)

with open(RESULT_DIR / "classification_report_v5.txt", "w") as f:
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
    f.write(report)

cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

plt.figure(figsize=(11, 9))
plt.imshow(cm, interpolation="nearest")
plt.title("EfficientNetB0 V5 - Confusion Matrix")
plt.colorbar()
ticks = np.arange(num_classes)
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(
    RESULT_DIR / "confusion_matrix_v5.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nEvaluation complete.")
print("Report:", (RESULT_DIR / "classification_report_v5.txt").resolve())
print("Confusion matrix:", (RESULT_DIR / "confusion_matrix_v5.png").resolve())
