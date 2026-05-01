"""
Residual CNN: skip connections inspired by ResNet.

Why this beats deep_cnn:
  - Skip connections let gradients flow directly to early layers — training
    a 6-conv network without them causes vanishing gradients and slow
    convergence. With them, every layer learns *residuals* (small
    corrections) rather than full transformations, which is far easier.
  - Uses the Functional API instead of Sequential so we can branch and
    merge tensors. The structure is identical to a mini ResNet-style block.
  - 1x1 "projection" convolutions on the skip path align channel counts
    when the number of filters changes between blocks.
  - GlobalAveragePooling at the end keeps the param count low (~150k total)
    while still capturing spatial features.

Block layout:
    Input → [Block1: 32ch] → [Block2: 64ch] → [Block3: 128ch] → GAP → Dense(10)
    Each block: Conv-BN-ReLU → Conv-BN → (+skip) → ReLU → MaxPool → Dropout
Usage:
    python train.py --model residual_cnn --epochs 20
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def residual_block(x, filters, stride=1):
    """
    One residual block:
      - Two 3x3 convs with BN (pre-activation style).
      - Skip connection with optional 1x1 projection if filters change.
    """
    shortcut = x

    # Main path
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut: align channels if they changed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, (1, 1), padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_model(input_shape=(28, 28, 1), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)

    # Stem: single conv to get into feature space
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Block 1 — 32 filters, 28x28
    x = residual_block(x, filters=32)
    x = layers.MaxPooling2D((2, 2))(x)   # → 14x14
    x = layers.Dropout(0.2)(x)

    # Block 2 — 64 filters, 14x14
    x = residual_block(x, filters=64)
    x = layers.MaxPooling2D((2, 2))(x)   # → 7x7
    x = layers.Dropout(0.2)(x)

    # Block 3 — 128 filters, 7x7
    x = residual_block(x, filters=128)
    x = layers.Dropout(0.3)(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="Residual_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
