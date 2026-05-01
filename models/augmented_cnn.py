"""
Augmented CNN: Simple CNN + built-in data augmentation.

Why this beats simple_cnn:
  - RandomRotation / RandomTranslation / RandomZoom are applied ON THE FLY
    during training only (Keras automatically disables them at inference).
    This gives the model effectively infinite training variety from the same
    60k images, which is the single biggest win on MNIST for a small net.
  - Padding="same" throughout keeps spatial dims consistent across blocks
    and squeezes a few extra accuracy points vs "valid" padding.
  - Added a second Dense(128) head — the tiny augmented model can absorb
    more capacity in the classifier since augmentation handles generalization.
Usage:
    python train.py --model augmented_cnn --epochs 20
"""

from tensorflow.keras import layers, models
import tensorflow as tf


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # --- Data augmentation (training only) ---
        layers.RandomRotation(0.08),          # ±~5° rotation
        layers.RandomTranslation(0.1, 0.1),   # ±10% shift in x and y
        layers.RandomZoom(0.1),               # ±10% zoom

        # --- Block 1 ---
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- Block 2 ---
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- Classifier head ---
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="Augmented_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
