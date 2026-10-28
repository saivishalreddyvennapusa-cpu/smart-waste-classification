
import os
from pathlib import Path

# Windows TensorFlow GPU DLL setup: must happen before importing TensorFlow.
if os.environ.get("CONDA_PREFIX"):
    _dll_handle = os.add_dll_directory(
        os.path.join(os.environ["CONDA_PREFIX"], "Library", "bin")
    )

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# -------------------- CONFIGURATION --------------------
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

DATA_DIR = Path("dataset/wastes")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

HEAD_WEIGHTS = MODEL_DIR / "waste_efficientnetb0_v5_head.weights.h5"
FINE_WEIGHTS = MODEL_DIR / "waste_efficientnetb0_v5_finetuned.weights.h5"
FINAL_WEIGHTS = MODEL_DIR / "waste_efficientnetb0_v5.weights.h5"

tf.keras.utils.set_random_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# -------------------- CHECK DATASET --------------------
if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
    raise FileNotFoundError(
        "Dataset folders not found. Expected:\n"
        f"{TRAIN_DIR.resolve()}\n{TEST_DIR.resolve()}"
    )

# -------------------- LOAD DATA --------------------
print("\nLoading training and validation datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    validation_split=0.15,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

if num_classes != 9:
    raise ValueError(
        f"Expected 9 classes, but found {num_classes}: {class_names}"
    )

print("Classes:", class_names)
print("Number of classes:", num_classes)

# Keep image values in 0-255 range. EfficientNetB0 includes
# its own input rescaling/preprocessing layers.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -------------------- CLASS WEIGHTS --------------------
# Calculate weights from training labels to help with imbalance.
train_labels = np.concatenate(
    [labels.numpy() for _, labels in train_ds], axis=0
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)
class_weights = {
    i: float(weights[i]) for i in range(num_classes)
}
print("Class weights:", class_weights)

# -------------------- BUILD MODEL --------------------
augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.10),
    ],
    name="augmentation"
)

# No 'name' argument: compatible with TensorFlow 2.10.1.
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

# Save only weights. This avoids serializing the complete model
# configuration, which triggered your previous error.
class SafeWeightsCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = str(filepath)
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get("val_accuracy")

        if value is not None and np.isfinite(value):
            if value > self.best:
                self.best = float(value)
                self.model.save_weights(self.filepath)
                print(
                    f"\nSaved best weights: {self.filepath} "
                    f"(val_accuracy={self.best:.4f})"
                )

def make_callbacks(filepath, patience=4):
    return [
        SafeWeightsCheckpoint(filepath),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

# -------------------- PHASE 1: TRAIN CLASSIFIER HEAD --------------------
print("\nPHASE 1: Training classifier head...")
compile_model(1e-3)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(HEAD_WEIGHTS, patience=4)
)

if not HEAD_WEIGHTS.exists():
    raise RuntimeError("Phase 1 did not save any checkpoint weights.")

model.load_weights(str(HEAD_WEIGHTS))
_, head_val_acc = model.evaluate(val_ds, verbose=1)
print(f"Head validation accuracy: {head_val_acc:.4f}")

# -------------------- PHASE 2: FINE-TUNE --------------------
print("\nPHASE 2: Fine-tuning EfficientNetB0...")

base_model.trainable = True

# Unfreeze the final 30 layers while keeping BatchNorm frozen.
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

compile_model(1e-5)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(FINE_WEIGHTS, patience=5)
)

if not FINE_WEIGHTS.exists():
    raise RuntimeError("Fine-tuning did not save checkpoint weights.")

# -------------------- SELECT BEST MODEL USING VALIDATION --------------------
print("\nSelecting best model using validation accuracy...")

# Fine-tuning callback restored the best fine-tuned weights.
# Evaluate the current model first.
_, fine_val_acc = model.evaluate(val_ds, verbose=1)
print(f"Fine-tuned validation accuracy: {fine_val_acc:.4f}")

# Load the head-only model in a separate, freshly built model
# with the backbone frozen, matching the Phase 1 architecture.
head_base = EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=IMG_SIZE + (3,)
)
head_base.trainable = False

head_inputs = layers.Input(shape=IMG_SIZE + (3,))
head_x = augmentation(head_inputs)
head_x = head_base(head_x, training=False)
head_x = layers.GlobalAveragePooling2D()(head_x)
head_x = layers.Dropout(0.30)(head_x)
head_x = layers.Dense(256, activation="relu")(head_x)
head_x = layers.Dropout(0.25)(head_x)
head_outputs = layers.Dense(num_classes, activation="softmax")(head_x)
head_model = models.Model(head_inputs, head_outputs)

head_model.load_weights(str(HEAD_WEIGHTS))
_, head_val_acc = head_model.evaluate(val_ds, verbose=1)
print(f"Head-only validation accuracy: {head_val_acc:.4f}")

if fine_val_acc >= head_val_acc:
    print("Selected fine-tuned model.")
    model.save_weights(str(FINAL_WEIGHTS))
else:
    print("Selected head-only model.")
    head_model.save_weights(str(FINAL_WEIGHTS))
    model = head_model

print("Final weights saved to:", FINAL_WEIGHTS)
model.load_weights(str(HEAD_WEIGHTS))
_, head_val_acc = model.evaluate(val_ds, verbose=0)

model.load_weights(str(FINE_WEIGHTS))
_, fine_val_acc = model.evaluate(val_ds, verbose=0)

if fine_val_acc >= head_val_acc:
    print("\nSelected fine-tuned model.")
    model.load_weights(str(FINE_WEIGHTS))
else:
    print("\nSelected classifier-head model.")
    model.load_weights(str(HEAD_WEIGHTS))

model.save_weights(str(FINAL_WEIGHTS))
print("Final weights saved to:", FINAL_WEIGHTS)

# -------------------- SAVE TRAINING HISTORY --------------------
history = {}
for key in history1.history:
    history[key] = list(history1.history[key])

for key, values in history2.history.items():
    history.setdefault(key, []).extend(list(values))

with open(RESULT_DIR / "training_history_v5.json", "w") as f:
    json.dump(history, f, indent=2)

for metric, filename, ylabel in [
    ("accuracy", "accuracy_v5.png", "Accuracy"),
    ("loss", "loss_v5.png", "Loss")
]:
    plt.figure()
    plt.plot(history.get(metric, []), label="Training")
    plt.plot(history.get("val_" + metric, []), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title("EfficientNetB0 V5 - " + ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=150)
    plt.close()

# -------------------- LOAD TEST DATA --------------------
print("\nLoading independent test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

if test_ds.class_names != class_names:
    raise ValueError(
        f"Train/test class mismatch.\n"
        f"Train: {class_names}\nTest: {test_ds.class_names}"
    )

test_ds = test_ds.prefetch(AUTOTUNE)

print("\nFINAL TEST EVALUATION")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds], axis=0
)
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

cm = confusion_matrix(
    y_true, y_pred, labels=np.arange(num_classes)
)

plt.figure(figsize=(11, 9))
plt.imshow(cm, interpolation="nearest")
plt.title("EfficientNetB0 V5 - Confusion Matrix")
plt.colorbar()
positions = np.arange(num_classes)
plt.xticks(positions, class_names, rotation=45, ha="right")
plt.yticks(positions, class_names)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(
    RESULT_DIR / "confusion_matrix_v5.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nTRAINING AND EVALUATION COMPLETE.")
print("Final weights:", FINAL_WEIGHTS)
print("Report:", RESULT_DIR / "classification_report_v5.txt")
print("Confusion matrix:", RESULT_DIR / "confusion_matrix_v5.png")
print("Training plots and history saved in:", RESULT_DIR.resolve())
