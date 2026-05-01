"""
Separable CNN: efficient convolutional model using depthwise separable convolutions.

Hyperparameter choices:
  - SeparableConv2D layers: reduce parameter count while retaining strong feature extraction.
  - BatchNorm after each conv: stabilizes training and improves convergence.
  - MaxPooling after conv blocks: progressively downsamples spatial dimensions.
  - Dropout (0.5 before output): helps reduce overfitting in dense layers.
  - Flatten + Dense(128): learns high-level feature combinations for classification.
  - Adam with lr=1e-3: effective optimizer for fast and stable training.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters):
    shortcut = x
    
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def build_model(input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Data augmentation inside model
    x = layers.RandomRotation(0.1)(inputs)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    x = residual_block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = residual_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = residual_block(x, 128)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="Separable_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    return model