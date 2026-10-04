"""Model factory for MalariaAI.

All models:
- accept (224, 224, 3) RGB images
- output one sigmoid probability: P(Parasitized)
- expect preprocessing/scaling to be handled by the data pipeline
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_custom_cnn(
    input_shape=(224, 224, 3),
    dropout=0.30,
) -> keras.Model:
    """Build the baseline custom CNN."""

    inputs = keras.Input(shape=input_shape, name="image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="parasitized_probability",
    )(x)

    return keras.Model(inputs, outputs, name="custom_cnn")


def build_mobilenetv2(
    input_shape=(224, 224, 3),
    trainable=False,
) -> keras.Model:
    """Build MobileNetV2 transfer-learning model."""

    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    base.trainable = trainable

    inputs = keras.Input(shape=input_shape, name="image")

    # MobileNetV2 expects [-1, 1].
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="parasitized_probability",
    )(x)

    return keras.Model(inputs, outputs, name="mobilenetv2")


def build_efficientnetb0(
    input_shape=(224, 224, 3),
    trainable=False,
) -> keras.Model:
    """Build EfficientNetB0 transfer-learning model."""

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    base.trainable = trainable

    inputs = keras.Input(shape=input_shape, name="image")

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="parasitized_probability",
    )(x)

    return keras.Model(inputs, outputs, name="efficientnetb0")


def build_model(model_name: str, input_shape=(224, 224, 3)) -> keras.Model:
    """Build a model by configuration name."""

    if model_name == "custom_cnn":
        return build_custom_cnn(input_shape)

    if model_name == "mobilenetv2":
        return build_mobilenetv2(input_shape)

    if model_name == "efficientnetb0":
        return build_efficientnetb0(input_shape)

    if model_name == "baseline_ref":
        # Keep the reference baseline separate later if its architecture
        # needs to reproduce a published/reference implementation exactly.
        return build_custom_cnn(input_shape)

    raise ValueError(
        f"Unknown model_name={model_name!r}. "
        "Expected one of: custom_cnn, mobilenetv2, "
        "efficientnetb0, baseline_ref."
    )
