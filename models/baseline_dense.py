"""
Baseline: fully-connected network. No convolutions.

Purpose: sanity-check that CNNs actually buy you something. If a dense
net gets 97% and your fancy CNN gets 97.2%, the complexity isn't earning
its keep.
"""

from tensorflow.keras import layers, models


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="Baseline_Dense")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
