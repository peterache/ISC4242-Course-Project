"""
Simple CNN: 2 conv blocks + dense head.

Hyperparameter choices:
  - 3x3 kernels: standard for small images.
  - 32 -> 64 filters: classic doubling as spatial dims shrink.
  - MaxPooling 2x2: halves spatial dims, reduces params.
  - Single dense layer: keeps it lightweight.
"""

from tensorflow.keras import layers, models


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="Simple_CNN")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
