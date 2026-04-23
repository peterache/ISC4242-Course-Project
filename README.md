# MNIST CNN Comparison

Course project: building and comparing CNN architectures on MNIST.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train a model (results go to `results/<model_name>/`):

```bash
python train.py --model simple_cnn
python train.py --model deep_cnn --epochs 20
```

Available models: `baseline_dense`, `simple_cnn`, `regularized_cnn`, `deep_cnn`.

Regenerate plots for an already-trained model without retraining:

```bash
python evaluate.py --model simple_cnn
```

Compare all trained models:

```bash
python compare.py
```

This reads every `results/*/metrics.json` and prints a sorted table plus
saves `results/comparison.csv` and `results/comparison.png`.

## Adding a new model

1. Create `models/your_model.py` with a `build_model()` function that
   returns a compiled `keras.Model`. Copy `models/simple_cnn.py` as a
   starting template.
2. Register it in `models/__init__.py`:
   ```python
   from . import your_model
   MODELS = {..., "your_model": your_model.build_model}
   ```
3. Train it: `python train.py --model your_model`

## Team workflow

Each teammate works on their own architecture in a separate branch.

```bash
git checkout -b alice/wide-cnn
# add models/wide_cnn.py, register it, train it
python train.py --model wide_cnn
git add models/wide_cnn.py models/__init__.py results/wide_cnn/
git commit -m "Add wide CNN variant"
git push origin alice/wide-cnn
```

Open a PR. Since each model is in its own file and each result folder is
per-model, parallel work doesn't conflict.

Note: `model.keras` weight files are `.gitignore`'d (too large). Plots
and `metrics.json` are committed so anyone can see results without
retraining.

## Project structure

```
ISC4242-Course-Project/
├── data.py                 shared MNIST loading + preprocessing
├── train.py                train one model, save all artifacts
├── evaluate.py             re-evaluate without retraining
├── compare.py              build comparison table across all models
├── requirements.txt
├── models/
│   ├── __init__.py         model registry
│   ├── baseline_dense.py
│   ├── simple_cnn.py
│   ├── regularized_cnn.py
│   └── deep_cnn.py
└── results/                created by train.py
    └── <model_name>/
        ├── metrics.json
        ├── history.csv
        ├── training_curves.png
        ├── confusion_matrix.png
        ├── classification_report.txt
        ├── misclassified.png
        └── model.keras     (gitignored)
```
