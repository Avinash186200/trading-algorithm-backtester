"""Classifier factory. The scaler lives inside each pipeline, so it is fitted on training rows only."""

from __future__ import annotations

import warnings

# SVC(probability=True) is deprecated in newer scikit-learn but still works; keep output readable.
warnings.filterwarnings("ignore", message=".*probability.*deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Maximum iterations.*", module="sklearn.neural_network")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("logreg", "rf", "svm", "mlp")

# Small grids on purpose: each one is searched with time-ordered CV on training data only.
GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"clf__C": [0.1, 1.0]},
    "rf": {"clf__max_depth": [3, 5], "clf__min_samples_leaf": [20, 50]},
    "svm": {"clf__C": [0.3, 1.0], "clf__gamma": ["scale"]},
    "mlp": {"clf__alpha": [0.1, 1.0, 3.0]},
}


def make_pipeline(name: str, seed: int = 0) -> Pipeline:
    """Return an unfitted scaler + classifier pipeline."""
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif name == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=seed)
    elif name == "mlp":
        # The original 8-4-2 shape with early stopping collapsed to a constant
        # prediction on some seeds: with ~600 training rows per group the 10%
        # (randomly drawn) validation split is ~60 rows, training stopped after a
        # few epochs, and the 2-unit bottleneck had not learned anything yet.
        # A 16-8 net without early stopping and with a stronger L2 grid (alpha
        # tuned by time-ordered CV) gives non-constant predictions on every seed
        # I checked (1, 2, 3, 7). Some grid points hit max_iter before full
        # convergence; that warning is silenced below because the heavy L2
        # penalty, not the iteration count, is what regularises the fit.
        clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def tuned_model(name: str, seed: int = 0, n_splits: int = 4) -> GridSearchCV:
    """Grid search with time-ordered folds (no shuffling across time)."""
    return GridSearchCV(make_pipeline(name, seed), GRIDS[name],
                        cv=TimeSeriesSplit(n_splits=n_splits), scoring="neg_log_loss")
