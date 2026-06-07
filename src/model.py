"""Train, persist, and load the gradient-boosted classifier.

The model predicts P(forward return > threshold) for a single stock-day given its
technical features. We use sklearn's HistGradientBoostingClassifier -- a fast,
LightGBM-style histogram gradient booster that ships with scikit-learn, so there
are no native-build dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_COLUMNS

log = logging.getLogger(__name__)


@dataclass
class TradeModel:
    """Wraps the fitted estimator plus the feature list it was trained on."""

    estimator: HistGradientBoostingClassifier
    feature_columns: list[str]
    horizon: int
    threshold: float

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of an up-move for each row of `features`."""
        X = features[self.feature_columns].to_numpy()
        return self.estimator.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("Saved model to %s", path)

    @staticmethod
    def load(path: str | Path) -> "TradeModel":
        return joblib.load(path)


def train_model(
    panel: pd.DataFrame,
    horizon: int,
    threshold: float,
    params: dict,
) -> TradeModel:
    """Fit the classifier on a panel produced by features.build_dataset."""
    X = panel[FEATURE_COLUMNS].to_numpy()
    y = panel["label"].to_numpy().astype(int)

    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X, y)

    base_rate = y.mean()
    log.info(
        "Trained on %d samples | up-move base rate %.3f | train accuracy %.3f",
        len(y),
        base_rate,
        clf.score(X, y),
    )
    return TradeModel(
        estimator=clf,
        feature_columns=list(FEATURE_COLUMNS),
        horizon=horizon,
        threshold=threshold,
    )
