"""Feature engineering — cœur pur (numpy/pandas, zéro I/O, zéro torch)."""

from .pipeline import build_features
from .scaler import FeatureScaler

__all__ = ["FeatureScaler", "build_features"]
