"""Scikit-learn Random Forest baseline for ECG classification."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class ECGBaselineRF(BaseEstimator):
    """
    Random Forest baseline for ECG beat classification.

    Operates on handcrafted morphology and interval features.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Args:
            n_estimators: Number of trees in the forest.
            random_state: Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ECGBaselineRF":
        """
        Train the Random Forest.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, n_classes).
        """
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy score.

        Args:
            X: Feature array.
            y: True labels.

        Returns:
            Accuracy (macro-averaged F1 preferred in reporting).
        """
        return self.model.score(X, y)


if __name__ == "__main__":
    print("Baseline model module. Use via training scripts.")
