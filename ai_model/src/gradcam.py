from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from .preprocessing import decode_resize_image


def find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Return the deepest convolution-like layer, including inside nested models."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                pass
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer
    raise ValueError("No convolutional layer found for Grad-CAM.")


def make_gradcam_heatmap(image_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str | None = None, pred_index: int | None = None):
    if last_conv_layer_name is None:
        last_conv_layer = find_last_conv_layer(model)
    else:
        last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_path: str | Path, heatmap: np.ndarray, output_path: str | Path, alpha: float = 0.35):
    original = Image.open(original_path).convert("RGB")
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original.size)
    cmap = plt.get_cmap("jet")
    colored = cmap(np.asarray(heatmap_img) / 255.0)[:, :, :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    overlay = Image.blend(original, colored, alpha=alpha)
    overlay.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for one image.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    image = decode_resize_image(tf.constant(args.image_path), args.image_size)
    image_batch = tf.expand_dims(image, axis=0)
    heatmap = make_gradcam_heatmap(image_batch, model)
    overlay_heatmap(args.image_path, heatmap, args.output_path)
    print(f"Grad-CAM saved to {args.output_path}")


if __name__ == "__main__":
    main()
