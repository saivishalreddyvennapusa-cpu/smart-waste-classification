
import os

# Add the Conda environment's CUDA DLL folder
os.add_dll_directory(
    os.path.join(os.environ["CONDA_PREFIX"], "Library", "bin")
)

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# Perform a small calculation on the GPU
with tf.device("/GPU:0"):
    a = tf.random.normal((1000, 1000))
    b = tf.random.normal((1000, 1000))
    c = tf.matmul(a, b)

print("GPU calculation completed:", c.shape)
