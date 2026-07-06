"""Resampling et normalisation timezone des données OHLCV."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_RESAMPLE_RULES: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """Rééchantillonne un DataFrame OHLCV vers un timeframe plus large.

    Args:
        df: DataFrame source avec index DatetimeIndex et colonnes OHLCV.
        target_timeframe: Timeframe cible parmi {"5m","15m","1h","4h","1d"}.

    Returns:
        DataFrame rééchantillonné, lignes avec NaN supprimées.

    Raises:
        KeyError: Si target_timeframe n'est pas supporté.
    """
    if target_timeframe not in _RESAMPLE_RULES:
        raise KeyError(
            f"Timeframe {target_timeframe!r} non supporté. "
            f"Choix : {list(_RESAMPLE_RULES)}"
        )
    rule = _RESAMPLE_RULES[target_timeframe]
    resampled = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["close"])
    logger.debug("Resample → %s : %d bougies", target_timeframe, len(resampled))
    return resampled


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Garantit que l'index du DataFrame est en UTC."""
    dti = pd.DatetimeIndex(df.index)
    if dti.tz is None:
        df = df.copy()
        df.index = dti.tz_localize("UTC")
        return df
    if str(dti.tz) != "UTC":
        df = df.copy()
        df.index = dti.tz_convert("UTC")
    return df
