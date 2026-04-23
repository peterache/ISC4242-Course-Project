"""
Train a model and save everything to results/<model_name>/.

Usage:
    python train.py --model simple_cnn
    python train.py --model deep_cnn --epochs 20 --batch-size 256

Saves:
    results/<model>/model.keras          trained weights
    results/<model>/history.csv          per-epoch loss/accuracy
    results/<model>/metrics.json         final test metrics
    results/<model>/training_curves.png  loss + accuracy plots
    results/<model>/confusion_matrix.png
    results/<model>/misclassified.png    sample of wrong predictions
    results/<model>/classification_report.txt
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report

from data import load_mnist, train_val_split
from models import get_model, MODELS

SEED = 42


def train_one_model(model_name, epochs, batch_size, output_root="results"):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    out_dir = Path(output_root) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # --- Load data ---
    print("Loading MNIST...")
    data = load_mnist()
    x_tr, y_tr, x_val, y_val = train_val_split(data, val_fraction=0.1)
    print(f"  Train: {x_tr.shape}, Val: {x_val.shape}, "
          f"Test: {data['x_test'].shape}")

    # --- Build model ---
    print(f"Building {model_name}...")
    model = get_model(model_name)
    model.summary()

    # --- Train ---
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6,
        ),
    ]
    print(f"Training for up to {epochs} epochs...")
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=cbs, verbose=2,
    )

    # --- Save artifacts ---
    save_history(history, out_dir)
    save_training_curves(history, model_name, out_dir)

    # --- Evaluate on test set ---
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(
        data["x_test"], data["y_test"], verbose=0
    )
    y_pred = np.argmax(model.predict(data["x_test"], verbose=0), axis=1)
    y_true = data["y_test_raw"]

    save_metrics(model, test_loss, test_acc, out_dir)
    save_confusion_matrix(y_true, y_pred, model_name, out_dir)
    save_classification_report(y_true, y_pred, out_dir)
    save_misclassified(data["x_test"], y_true, y_pred, out_dir)

    # --- Save model weights ---
    model.save(out_dir / "model.keras")
    print(f"\nAll results saved to {out_dir}/")
    print(f"Test accuracy: {test_acc:.4f}")


def save_history(history, out_dir):
    df = pd.DataFrame(history.history)
    df.index.name = "epoch"
    df.to_csv(out_dir / "history.csv")


def save_training_curves(history, model_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set(title=f"{model_name} — Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Validation")
    axes[1].set(title=f"{model_name} — Accuracy",
                xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close()


def save_metrics(model, test_loss, test_acc, out_dir):
    metrics = {
        "model_name": model.name,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "total_params": int(model.count_params()),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def save_confusion_matrix(y_true, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close()


def save_classification_report(y_true, y_pred, out_dir):
    report = classification_report(y_true, y_pred, digits=4)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)


def save_misclassified(x_test, y_true, y_pred, out_dir, n=15):
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        return
    rng = np.random.default_rng(SEED)
    sample = rng.choice(wrong_idx, size=min(n, len(wrong_idx)), replace=False)

    cols = 5
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, sample):
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        ax.set_title(f"T:{y_true[i]} P:{y_pred[i]}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "misclassified.png", dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument(
        "--model", required=True, choices=list(MODELS.keys()),
        help="Which model to train.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()

    train_one_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
