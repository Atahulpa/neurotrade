"""Réseau 1D-CNN pour séries temporelles.

Entrée : (n_samples, seq_len, n_features)
       → permutée en (n_samples, n_features, seq_len) pour Conv1d.
Sortie : (n_samples, 3) — probabilités pour {-1, 0, +1}.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurotrade.config.schema import ModelConfig

from .base import BaseModel


class CNN1D(BaseModel):
    """1D-CNN : convolutions temporelles + pooling adaptatif + tête de classification."""

    def __init__(self, n_features: int, config: ModelConfig) -> None:
        super().__init__()
        self._config = config
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x_perm: torch.Tensor = x.permute(0, 2, 1)
        x_conv: torch.Tensor = self.conv(x_perm).squeeze(-1)  # → (batch, 64)
        result: torch.Tensor = self.head(x_conv)
        return result

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> CNN1D:
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
