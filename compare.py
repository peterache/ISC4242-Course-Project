"""
Collect metrics.json from every subfolder of results/ and print a
comparison table. Also writes a summary CSV and a bar chart.

Usage:
    python compare.py
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main(results_root="results"):
    root = Path(results_root)
    if not root.exists():
        print(f"No {results_root}/ directory found.")
        return

    rows = []
    for metrics_file in sorted(root.glob("*/metrics.json")):
        with open(metrics_file) as f:
            rows.append(json.load(f))

    if not rows:
        print(f"No metrics.json files found under {results_root}/.")
        return

    df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    df = df[["model_name", "test_accuracy", "test_loss", "total_params"]]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    df.to_csv(root / "comparison.csv", index=False)
    print(f"Saved {root / 'comparison.csv'}")

    # Bar chart: accuracy by model
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["model_name"], df["test_accuracy"])
    ax.set_xlabel("Test accuracy")
    ax.set_title("Model comparison - test accuracy")
    ax.set_xlim(
        max(0, df["test_accuracy"].min() - 0.02),
        min(1.0, df["test_accuracy"].max() + 0.01),
    )
    for i, v in enumerate(df["test_accuracy"]):
        ax.text(v, i, f" {v:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(root / "comparison.png", dpi=120)
    print(f"Saved {root / 'comparison.png'}")


if __name__ == "__main__":
    main()
