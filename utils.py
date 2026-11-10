import io

import numpy as np
import tensorflow as tf
import matplotlib
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224


def pretty(name):
    """'Tomato___Early_blight' -> 'Tomato - Early blight'"""
    crop, _, condition = name.partition("___")
    clean = lambda s: s.replace("_", " ").replace("(", "").replace(")", "").strip()
    return f"{clean(crop)} - {clean(condition)}"


def decode(raw_bytes):
    """Decode an uploaded file to an RGB uint8 array.
    Uses TensorFlow's decoder first, the same one used when the model was trained and
    evaluated, so predictions match the evaluation pipeline. Falls back to PIL for
    formats TensorFlow cannot read (e.g. CMYK JPEGs)."""
    try:
        return tf.io.decode_image(raw_bytes, channels=3, expand_animations=False).numpy()
    except Exception:
        return np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))


def prepare(img_uint8):
    """Resize exactly like Keras' image_dataset_from_directory (bilinear). Returns a
    float32 batch with values in 0-255; the model applies MobileNetV2 scaling itself."""
    resized = tf.image.resize(img_uint8[None, ...], (IMG_SIZE, IMG_SIZE), method="bilinear")
    return resized.numpy().astype("float32")


def get_parts(model):
    """Layers needed for Grad-CAM (the saved model nests MobileNetV2 as one layer)."""
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    gap = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    drop = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dropout))
    dense = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dense))
    return base, gap, drop, dense


def gradcam(batch, parts):
    base, gap, drop, dense = parts
    x = preprocess_input(tf.cast(batch, tf.float32))
    with tf.GradientTape() as tape:
        conv_out = base(x, training=False)
        tape.watch(conv_out)
        preds = dense(drop(gap(conv_out), training=False))
        idx = tf.argmax(preds[0])
        score = preds[:, idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
    cam = cam / tf.maximum(tf.reduce_max(cam), 1e-30)   # scale to 0-1; guard against an all-zero map
    return cam.numpy()


def overlay(batch, cam, alpha=0.4):
    img = np.clip(batch[0], 0, 255).astype("uint8")
    cam_img = Image.fromarray(np.uint8(255 * cam)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    heat = matplotlib.colormaps["jet"](np.array(cam_img) / 255.0)[..., :3]
    return np.uint8((1 - alpha) * img + alpha * 255 * heat)
