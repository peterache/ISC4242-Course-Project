"""
Re-generate plots and metrics for an already-trained model.

Usage:
    python evaluate.py --model simple_cnn

Useful if:
  - You lost a plot and don't want to retrain.
  - You want to inspect a teammate's model from their branch.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import load_mnist
from train import (
    save_confusion_matrix,
    save_classification_report,
    save_misclassified,
    save_metrics,
)
from models import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / args.model
    weights_path = out_dir / "model.keras"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No trained model at {weights_path}. Run train.py first."
        )

    print(f"Loading {weights_path}...")
    model = tf.keras.models.load_model(weights_path)
    data = load_mnist()

    test_loss, test_acc = model.evaluate(
        data["x_test"], data["y_test"], verbose=0
    )
    y_pred = np.argmax(model.predict(data["x_test"], verbose=0), axis=1)
    y_true = data["y_test_raw"]

    save_metrics(model, test_loss, test_acc, out_dir)
    save_confusion_matrix(y_true, y_pred, args.model, out_dir)
    save_classification_report(y_true, y_pred, out_dir)
    save_misclassified(data["x_test"], y_true, y_pred, out_dir)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Regenerated artifacts in {out_dir}/")


if __name__ == "__main__":
    main()
