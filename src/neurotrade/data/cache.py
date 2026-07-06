"""Cache parquet pour les séries OHLCV."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from neurotrade.config.schema import DataConfig

logger = logging.getLogger(__name__)


def cache_path(config: DataConfig) -> Path:
    """Retourne le chemin canonique du fichier parquet pour cette config."""
    symbol_safe = config.symbol.replace("/", "_")
    filename = f"{symbol_safe}_{config.timeframe}_{config.start}_{config.end}.parquet"
    return Path(config.cache_dir) / filename


def save_to_cache(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un DataFrame OHLCV en parquet (pyarrow)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.info("Cache sauvegardé : %s (%d bougies)", path, len(df))


def load_from_cache(path: Path) -> pd.DataFrame:
    """Charge un DataFrame OHLCV depuis le cache parquet.

    Returns:
        DataFrame avec index DatetimeIndex UTC et colonnes float64.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cache introuvable : {path}")
    df: pd.DataFrame = pd.read_parquet(path)
    # Garantit l'index UTC (pyarrow peut perdre la timezone selon la version)
    dti = pd.DatetimeIndex(df.index)
    if dti.tz is None:
        df = df.copy()
        df.index = dti.tz_localize("UTC")
    logger.info("Cache chargé : %s (%d bougies)", path, len(df))
    return df


def load_or_fetch(
    config: DataConfig,
    fetch_fn: Callable[[DataConfig], pd.DataFrame],
) -> pd.DataFrame:
    """Charge depuis le cache si dispo, sinon télécharge et met en cache.

    Args:
        config: Configuration de la source de données.
        fetch_fn: Callable (DataConfig) → pd.DataFrame (typiquement fetch_ohlcv).

    Returns:
        DataFrame OHLCV propre.
    """
    path = cache_path(config)
    if path.exists():
        logger.info("Cache hit : %s", path)
        return load_from_cache(path)

    logger.info("Cache miss — téléchargement en cours…")
    df = fetch_fn(config)
    save_to_cache(df, path)
    return df
