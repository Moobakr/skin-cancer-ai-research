from __future__ import annotations

import tensorflow as tf


def decode_resize_image(path: tf.Tensor, image_size: int = 224) -> tf.Tensor:
    """Read an image from disk and resize it.

    We keep pixel values as float32 in the [0, 255] range. EfficientNet in Keras
    includes its own preprocessing behavior, and the custom CNN adds a rescaling
    layer in the model.
    """
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    return image


def make_image_dataset(
    paths,
    labels,
    image_size: int,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = 42,
):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )

    def load(path, label):
        image = decode_resize_image(path, image_size)
        if augment:
            image = augmentation(image, training=True)
        return image, label

    # Bound decoded-image buffers on laptops with limited free RAM.
    return ds.map(load, num_parallel_calls=2).batch(batch_size).prefetch(1)


def make_multimodal_dataset(
    paths,
    metadata_array,
    labels,
    image_size: int,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = 42,
):
    ds = tf.data.Dataset.from_tensor_slices((paths, metadata_array.astype("float32"), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )

    def load(path, metadata, label):
        image = decode_resize_image(path, image_size)
        if augment:
            image = augmentation(image, training=True)
        return {"image_input": image, "metadata_input": metadata}, label

    return ds.map(load, num_parallel_calls=2).batch(batch_size).prefetch(1)
