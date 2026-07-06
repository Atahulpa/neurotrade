"""Boucle d'entraînement avec early stopping.

Sépare la logique d'entraînement de l'architecture du réseau.
Le Trainer est utilisé par le runner walk-forward pour entraîner
n'importe quel BaseModel via la même interface.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neurotrade.config.schema import ModelConfig

from .base import BaseModel

logger = logging.getLogger(__name__)

# Les labels {-1, 0, +1} sont réindexés en {0, 1, 2} pour CrossEntropy
_LABEL_OFFSET = 1


class Trainer:
    """Entraîne un BaseModel avec mini-batches, CrossEntropy et early stopping.

    Utilisation :
        trainer = Trainer(model, config)
        val_losses = trainer.train(X_train, y_train, X_val, y_val)
    """

    def __init__(self, model: BaseModel, config: ModelConfig) -> None:
        self.model = model
        self.config = config

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[float]:
        """Entraîne le modèle et retourne la liste des val_loss par epoch.

        Args:
            X_train: Features train.
            y_train: Labels train en {-1, 0, +1}.
            X_val: Features val.
            y_val: Labels val en {-1, 0, +1}.

        Returns:
            Historique de val_loss (un float par epoch, jusqu'à early stopping).
        """
        cfg = self.config
        device = torch.device("cpu")  # CPU par défaut (reproductibilité)

        # Conversion numpy → tensor, labels {-1,0,+1} → {0,1,2}
        X_tr = torch.from_numpy(X_train).float().to(device)
        y_tr = torch.from_numpy(y_train.astype(np.int64) + _LABEL_OFFSET).long().to(device)
        X_vl = torch.from_numpy(X_val).float().to(device)
        y_vl = torch.from_numpy(y_val.astype(np.int64) + _LABEL_OFFSET).long().to(device)

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        self.model.to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        val_losses: list[float] = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.max_epochs):
            self.model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_vl)
                val_loss = criterion(val_logits, y_vl).item()
            val_losses.append(val_loss)

            logger.debug("Epoch %d/%d — val_loss=%.4f", epoch + 1, cfg.max_epochs, val_loss)

            # Early stopping : on garde le meilleur modèle implicitement
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info(
                        "Early stopping à l'epoch %d (patience=%d, best_val=%.4f)",
                        epoch + 1, cfg.patience, best_val_loss,
                    )
                    break

        return val_losses
