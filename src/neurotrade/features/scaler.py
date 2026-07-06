"""Normalisation train-only des features (zéro fuite val→train).

Règle fondamentale : fit() ne doit être appelé QUE sur les données d'entraînement.
Si transform() est appelé avant fit(), une RuntimeError est levée — c'est intentionnel.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FeatureScaler:
    """StandardScaler adapté au walk-forward.

    fit() sur train uniquement → transform() sur train, val, test.
    L'invariant empêche physiquement d'appliquer une normalisation non calibrée.
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    def fit(self, X_train: np.ndarray) -> FeatureScaler:
        """Calcule μ et σ sur X_train uniquement.

        Args:
            X_train: Matrice (n_samples, n_features) de données d'entraînement.

        Returns:
            self (pour chaîner fit().transform()).
        """
        if X_train.shape[0] < 2:
            raise ValueError("X_train doit contenir au moins 2 échantillons.")
        self._mean = X_train.mean(axis=0)
        # ddof=1 pour l'estimateur sans biais
        self._std = X_train.std(axis=0, ddof=1)
        # Évite la division par zéro sur les features constantes
        self._std = np.where(self._std < 1e-8, 1.0, self._std)
        logger.debug(
            "Scaler calibré sur %d échantillons, %d features.", *X_train.shape
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applique la normalisation (μ, σ) apprise sur train.

        Args:
            X: Matrice (n_samples, n_features).

        Returns:
            Matrice normalisée de même forme.

        Raises:
            RuntimeError: Si fit() n'a pas été appelé.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "FeatureScaler.transform() appelé avant fit(). "
                "Appelez d'abord fit(X_train)."
            )
        result: np.ndarray = (X - self._mean) / self._std
        return result

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Raccourci fit + transform sur le même tableau (train uniquement)."""
        return self.fit(X_train).transform(X_train)
