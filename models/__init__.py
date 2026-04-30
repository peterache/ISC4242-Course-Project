"""
Model registry.

Each model module must expose a `build_model()` function that returns a
compiled keras.Model. To add a new model:

    1. Create models/your_model.py with a build_model() function.
    2. Add it to MODELS below.
    3. Run: python train.py --model your_model
"""

from . import baseline_dense, simple_cnn, regularized_cnn, deep_cnn, wide_cnn

MODELS = {
    "baseline_dense": baseline_dense.build_model,
    "simple_cnn": simple_cnn.build_model,
    "regularized_cnn": regularized_cnn.build_model,
    "deep_cnn": deep_cnn.build_model,
    "wide_cnn": wide_cnn.build_model
}


def get_model(name):
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODELS.keys())}"
        )
    return MODELS[name]()
