"""Unsupervised anomaly detectors trained on normal traffic only.

Every detector exposes the same interface:
    fit(X_normal)      learn what normal traffic looks like
    score(X)           anomaly score, higher = more anomalous
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM


def _subsample(X, max_rows, seed):
    if len(X) <= max_rows:
        return X
    idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
    return X[idx]


class IsolationForestDetector:
    name = "Isolation Forest"

    def __init__(self, seed=42):
        self.model = IsolationForest(n_estimators=200, max_samples=256,
                                     random_state=seed, n_jobs=-1)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class OneClassSVMDetector:
    name = "One-Class SVM"

    def __init__(self, seed=42, max_rows=10000):
        self.seed, self.max_rows = seed, max_rows
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)

    def fit(self, X):
        # Kernel SVMs scale quadratically, so train on a random subsample.
        self.model.fit(_subsample(X, self.max_rows, self.seed))
        return self

    def score(self, X):
        return -self.model.decision_function(X)


class LOFDetector:
    name = "Local Outlier Factor"

    def __init__(self, seed=42, max_rows=20000):
        self.seed, self.max_rows = seed, max_rows
        self.model = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)

    def fit(self, X):
        self.model.fit(_subsample(X, self.max_rows, self.seed))
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class AutoencoderDetector:
    """A dense autoencoder (input -> 32 -> 8 -> 32 -> input). Trained to
    reconstruct normal traffic; attacks reconstruct poorly, so the mean
    squared reconstruction error is the anomaly score."""

    name = "Autoencoder"

    def __init__(self, seed=42):
        self.model = MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation="relu",
                                  learning_rate_init=1e-3, batch_size=256,
                                  max_iter=100, early_stopping=True,
                                  n_iter_no_change=5, random_state=seed)

    def fit(self, X):
        self.model.fit(X, X)
        return self

    def score(self, X):
        return np.mean((X - self.model.predict(X)) ** 2, axis=1)


ALL_DETECTORS = {
    "iforest": IsolationForestDetector,
    "ocsvm": OneClassSVMDetector,
    "lof": LOFDetector,
    "autoencoder": AutoencoderDetector,
}
