"""Réseau MLP (Multi-Layer Perceptron) — architecture de base.

Entrée : (n_samples, n_features) — un vecteur de features par bougie.
Sortie : (n_samples, 3) — probabilités pour {-1, 0, +1}.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurotrade.config.schema import ModelConfig

from .base import BaseModel


class MLP(BaseModel):
    """MLP : couches linéaires + ReLU + dropout.

    Architecture : Linear(input→h0) → ReLU → Dropout → … → Linear(hN→3) → Softmax.
    """

    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in config.hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)
        self._config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> MLP:
        """Délègue l'entraînement au Trainer (appel depuis runner).

        Note : cette méthode est un raccourci autonome sans val set.
        Pour le walk-forward complet, utiliser Trainer.train().
        """
        from .trainer import Trainer

        X_val = X_train[-max(1, len(X_train) // 10) :]
        y_val = y_train[-max(1, len(y_train) // 10) :]
        X_t = X_train[: -max(1, len(X_train) // 10)]
        y_t = y_train[: -max(1, len(y_train) // 10)]
        trainer = Trainer(self, self._config)
        trainer.train(X_t, y_t, X_val, y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X).float()
            logits = self.forward(x_t)
            proba = F.softmax(logits, dim=1)
        result: np.ndarray = proba.numpy()
        return result
