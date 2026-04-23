"""
MNIST data loading and preprocessing.

Used by train.py and evaluate.py so preprocessing is consistent across
every model. Don't reimplement this in your model files.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def load_mnist():
    """
    Load MNIST and apply standard preprocessing.

    Steps:
      - Normalize pixel values from [0, 255] to [0, 1].
      - Add channel dim: (N, 28, 28) -> (N, 28, 28, 1).
      - One-hot encode labels (10 classes).

    Returns:
        dict with keys: x_train, y_train, x_test, y_test,
                        y_train_raw, y_test_raw (integer labels)
    """
    (x_train, y_train_raw), (x_test, y_test_raw) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes=10)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
    }


def train_val_split(data, val_fraction=0.1, seed=42):
    """Split the training set into train/val. Returns (x_tr, y_tr, x_val, y_val)."""
    rng = np.random.default_rng(seed)
    n = len(data["x_train"])
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return (
        data["x_train"][tr_idx], data["y_train"][tr_idx],
        data["x_train"][val_idx], data["y_train"][val_idx],
    )
