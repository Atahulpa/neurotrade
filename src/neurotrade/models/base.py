"""Interface abstraite commune à tous les modèles NeuroTrade.

Masque l'architecture derrière fit/predict_proba pour que backtest/
ne sache pas si c'est un MLP, un CNN ou un LSTM.

Contrat d'entrée :
- MLP     : X de forme (n_samples, n_features)
- CNN1D   : X de forme (n_samples, window_size, n_features)
- LSTMModel : X de forme (n_samples, window_size, n_features)

Les labels y sont des entiers {-1, 0, +1} réindexés en {0, 1, 2} en interne.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Interface commune fit / predict_proba."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> BaseModel:
        """Entraîne le modèle.

        Args:
            X_train: Features (n_samples, …).
            y_train: Labels entiers {-1, 0, +1}, forme (n_samples,).

        Returns:
            self.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités des 3 classes {-1, 0, +1}.

        Returns:
            Tableau (n_samples, 3) avec probabilités sommant à 1.
            Colonne 0 = P(label=-1), colonne 1 = P(label=0), colonne 2 = P(label=+1).
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne le label prédit (argmax sur predict_proba), en {-1, 0, +1}."""
        proba = self.predict_proba(X)
        argmax: np.ndarray = proba.argmax(axis=1)
        return argmax - 1  # {0,1,2} → {-1,0,+1}
