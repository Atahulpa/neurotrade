"""Pipeline de features : assemble toutes les features en une matrice numpy.

Ordre des colonnes (déterministe) :
  [ret_1, ret_5, ..., ret_H, vol, momentum, vol_rel*, log_vol*]
  (* uniquement si use_volume=True)

Ce module est un cœur pur : zéro I/O, zéro réseau, zéro torch.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from neurotrade.config.schema import FeaturesConfig

from .momentum import momentum_zscore
from .returns import log_returns
from .volatility import realized_volatility
from .volume import volume_features

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, config: FeaturesConfig) -> np.ndarray:
    """Construit la matrice de features à partir d'un DataFrame OHLCV.

    Ne fait aucun I/O. La normalisation n'est PAS appliquée ici (voir FeatureScaler).
    Les lignes avec NaN (début de série) sont supprimées.

    Args:
        df: DataFrame OHLCV avec colonnes [open, high, low, close, volume].
        config: Paramètres du pipeline.

    Returns:
        Matrice (n_samples, n_features) de type float64.

    Raises:
        ValueError: Si le nombre de features dépasse config.max_features.
    """
    parts: list[pd.DataFrame | pd.Series] = []

    # Log-returns multi-horizons
    parts.append(log_returns(df["close"], config.horizons))

    # Volatilité réalisée
    parts.append(realized_volatility(df["close"], config.vol_window).rename("vol"))

    # Z-score de momentum
    parts.append(momentum_zscore(df["close"], config.momentum_window).rename("momentum"))

    # Features volume (optionnel)
    if config.use_volume:
        parts.append(volume_features(df["volume"], window=config.vol_window))

    combined = pd.concat(parts, axis=1)

    n_features = combined.shape[1]
    if n_features > config.max_features:
        raise ValueError(
            f"Le pipeline produit {n_features} features > max_features={config.max_features}. "
            "Réduisez horizons ou désactivez use_volume."
        )

    # Supprime les lignes avec NaN (début de série : fenêtres pas encore remplies)
    combined = combined.dropna()
    logger.debug("Features : %d echantillons x %d features", *combined.shape)
    return combined.to_numpy(dtype=np.float64)


def feature_names(config: FeaturesConfig) -> list[str]:
    """Retourne les noms des colonnes dans l'ordre du pipeline."""
    names = [f"ret_{h}" for h in config.horizons]
    names.append("vol")
    names.append("momentum")
    if config.use_volume:
        names += ["vol_rel", "log_vol"]
    return names
