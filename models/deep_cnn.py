"""
Deep CNN: 3 conv blocks with L2 weight decay + GlobalAveragePooling.

Hyperparameter choices:
  - L2 regularization (1e-4): penalizes large weights.
  - GlobalAveragePooling instead of Flatten: fewer params, less overfitting.
  - 3 blocks: 32 -> 64 -> 128 filters.
  - Lower LR (5e-4) for stability with the deeper architecture.
"""

from tensorflow.keras import layers, models, regularizers
import tensorflow as tf


def build_model(input_shape=(28, 28, 1), num_classes=10):
    l2 = regularizers.l2(1e-4)

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=l2),
        layers.BatchNormalization(),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="Deep_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
