"""Architectures NN et entraînement — seule couche qui dépend de torch."""

from .base import BaseModel as NTBaseModel
from .mlp import MLP
from .trainer import Trainer

__all__ = ["MLP", "NTBaseModel", "Trainer"]
