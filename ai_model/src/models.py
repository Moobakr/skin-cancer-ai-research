from __future__ import annotations

import tensorflow as tf


def sparse_categorical_focal_loss(gamma: float = 2.0, label_smoothing: float = 0.1):
    """Focal loss with label smoothing for sparse integer labels.

    Focal loss downweights easy/majority-class predictions.
    Label smoothing prevents overconfident predictions on any single class.
    """
    def loss_fn(y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_onehot = tf.one_hot(y_true_int, tf.shape(y_pred)[-1])
        smoothed = y_true_onehot * (1.0 - label_smoothing) + label_smoothing / num_classes
        ce = -tf.reduce_sum(smoothed * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)), axis=-1)
        probs = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - probs, gamma)
        return focal_weight * ce
    return loss_fn


def build_custom_cnn(image_size: int, num_classes: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image_input")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(inputs, outputs, name="custom_cnn_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=sparse_categorical_focal_loss(gamma=2.0),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
    )
    return model


def efficientnet_feature_extractor(image_size: int, trainable: bool = False) -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = trainable
    return base


def build_efficientnet_classifier(
    image_size: int,
    num_classes: int,
    learning_rate: float = 1e-3,
    base_trainable: bool = False,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image_input")
    base = efficientnet_feature_extractor(image_size, trainable=base_trainable)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(inputs, outputs, name="efficientnetb3_image_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=sparse_categorical_focal_loss(gamma=2.0),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
    )
    return model


def build_multimodal_classifier(
    image_size: int,
    metadata_dim: int,
    num_classes: int,
    learning_rate: float = 1e-3,
    base_trainable: bool = False,
) -> tf.keras.Model:
    image_input = tf.keras.Input(shape=(image_size, image_size, 3), name="image_input")
    metadata_input = tf.keras.Input(shape=(metadata_dim,), name="metadata_input")

    base = efficientnet_feature_extractor(image_size, trainable=base_trainable)
    image_features = base(image_input, training=False)
    image_features = tf.keras.layers.GlobalAveragePooling2D(name="image_features")(image_features)
    image_features = tf.keras.layers.Dropout(0.35)(image_features)

    metadata_features = tf.keras.layers.Dense(64, activation="relu")(metadata_input)
    metadata_features = tf.keras.layers.BatchNormalization()(metadata_features)
    metadata_features = tf.keras.layers.Dropout(0.25)(metadata_features)
    metadata_features = tf.keras.layers.Dense(32, activation="relu")(metadata_features)

    x = tf.keras.layers.Concatenate(name="fusion_layer")([image_features, metadata_features])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(
        inputs={"image_input": image_input, "metadata_input": metadata_input},
        outputs=outputs,
        name="efficientnetb3_metadata_fusion_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=sparse_categorical_focal_loss(gamma=2.0),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
    )
    return model


def fine_tune_efficientnet(model: tf.keras.Model, learning_rate: float = 1e-5, unfreeze_last: int = 60) -> tf.keras.Model:
    """Unfreeze the last layers of the EfficientNet base for careful fine-tuning."""
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Could not find EfficientNet base model for fine-tuning.")

    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=sparse_categorical_focal_loss(gamma=2.0),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
    )
    return model
