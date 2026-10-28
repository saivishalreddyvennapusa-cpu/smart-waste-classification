# Smart Waste Classification Prototype (V5)

This is a separate prototype folder. It does not modify `train.py` or any existing V2/V3/V4 model.

## Features
- Streamlit web UI
- Upload JPG/PNG/WEBP image
- Capture a photo using the browser camera
- Prediction across the nine trained classes
- Top-three probabilities and disposal guidance
- Optional OpenCV USB-camera demo for a Raspberry Pi/Linux host

## Required folder layout

Keep this folder inside your existing project directory:

```
wasteclassifierproject/
├── models/
│   └── waste_efficientnetb0_v5_finetuned.weights.h5
├── dataset/
├── train.py
└── prototype/
    ├── app.py
    ├── embedded_camera.py
    └── requirements.txt
```

The checkpoint path is expected to be `../models/waste_efficientnetb0_v5_finetuned.weights.h5` relative to this folder.

## Windows: Streamlit app

Use the existing `(wastegpu)` Conda environment. In Anaconda Prompt:

```bat
conda activate wastegpu
cd C:\Users\saivi\OneDrive\Desktop\wasteclassifierproject\prototype
python -m pip install streamlit pillow
streamlit run app.py
```

Open the local URL printed by Streamlit. Select **Upload image** or **Take photo with camera**. Browser camera permission is required for photo capture. The model runs locally on the computer.

If Streamlit reports TensorFlow missing, run `python -c "import tensorflow as tf; print(tf.__version__)"` in the same prompt and verify it prints 2.10.1.

## Embedded prototype: Raspberry Pi + USB webcam

Recommended practical prototype: Raspberry Pi 4/5 (prefer 8 GB RAM) or a small Linux computer + USB webcam + display. EfficientNetB0 is too large for a typical ESP32-CAM to run directly. Do not expect this full model to run on an ESP32-CAM without model conversion, quantization and a much smaller architecture.

1. Copy the whole project or at least `prototype/` and `models/waste_efficientnetb0_v5_finetuned.weights.h5` to the Linux device, keeping the same relative layout.
2. Install a TensorFlow build compatible with that device's OS/Python, plus `opencv-python` and `numpy`. TensorFlow 2.10.1 Windows GPU setup is not portable to Raspberry Pi; the Pi will normally run CPU inference unless separately configured.
3. Connect a USB webcam.
4. From `prototype/`, run `python embedded_camera.py`.
5. Press SPACE to classify the current frame; press Q to quit.

Install dependencies suitable for the target's Python/OS. Do not blindly install the Windows `wastegpu` environment on the Pi.

## Important
- This model was evaluated at 86.40% test accuracy on the provided test split.
- Confidence is not a guarantee. Test with real objects, lighting, backgrounds and camera distances.
- Waste handling guidance is generic; follow local municipal rules. Batteries, e-waste and some bulbs may require special collection.
- The model uses the same 224x224 input, EfficientNetB0 backbone and 9-class output architecture as the training script described in the project context.
