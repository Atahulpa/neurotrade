"""Réseau LSTM pour séries temporelles.

Entrée : (n_samples, seq_len, n_features).
Sortie : (n_samples, 3) — probabilités pour {-1, 0, +1}.

Avantage du LSTM : capte les dépendances à long terme que le CNN ne voit pas.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurotrade.config.schema import ModelConfig

from .base import BaseModel


class LSTMModel(BaseModel):
    """LSTM à 2 couches + tête de classification."""

    def __init__(self, n_features: int, config: ModelConfig) -> None:
        super().__init__()
        self._config = config
        hidden = 64
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if config.dropout > 0 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last: torch.Tensor = out[:, -1, :]  # dernier pas de temps → (batch, hidden)
        result: torch.Tensor = self.head(last)
        return result

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> LSTMModel:
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
