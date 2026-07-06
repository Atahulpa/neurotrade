"""Téléchargement de données OHLCV via ccxt.

Ce module est le seul point d'entrée réseau du projet.
Les données sont paginées par lots de 1 000 bougies (limite Binance).
"""

from __future__ import annotations

import logging
from typing import Any

import ccxt
import pandas as pd

from neurotrade.config.schema import DataConfig

logger = logging.getLogger(__name__)

# Durée en millisecondes de chaque timeframe (pour calculer le pas entre pages)
_TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

_BATCH_SIZE = 1_000  # Binance limite à 1 000 bougies par requête


def fetch_ohlcv(config: DataConfig) -> pd.DataFrame:
    """Télécharge les données OHLCV pour la plage définie dans config.

    Pagination automatique : appelle l'exchange en boucle jusqu'à couvrir
    [config.start, config.end]. Rate-limiting géré par ccxt.

    Args:
        config: Section [data] de AppConfig.

    Returns:
        DataFrame avec index DatetimeIndex UTC et colonnes
        [open, high, low, close, volume], trié et dédupliqué.

    Raises:
        ValueError: Aucune donnée renvoyée par l'exchange.
    """
    exchange: Any = getattr(ccxt, config.exchange)({"enableRateLimit": True})

    tf_ms = _TIMEFRAME_MS[config.timeframe]
    since_ms: int = exchange.parse8601(config.start + "T00:00:00Z")
    end_ms: int = exchange.parse8601(config.end + "T23:59:59Z")

    rows: list[Any] = []
    current_ms = since_ms

    while current_ms <= end_ms:
        logger.info(
            "Fetch %s %s depuis %s…",
            config.symbol,
            config.timeframe,
            pd.Timestamp(current_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
        )
        batch: list[Any] = exchange.fetch_ohlcv(
            config.symbol, config.timeframe, since=current_ms, limit=_BATCH_SIZE
        )
        if not batch:
            break
        rows.extend(batch)
        last_ts: int = int(batch[-1][0])
        current_ms = last_ts + tf_ms
        # Arrêt anticipé si le batch est incomplet (fin de l'historique disponible)
        if len(batch) < _BATCH_SIZE:
            break

    if not rows:
        raise ValueError(
            f"Aucune donnée téléchargée pour {config.symbol} ({config.timeframe})"
            f" entre {config.start} et {config.end}."
        )

    df = _rows_to_dataframe(rows)
    # Tronque à la plage demandée (l'exchange peut renvoyer des bougies après end)
    end_cutoff = pd.Timestamp(config.end, tz="UTC") + pd.Timedelta(days=1)
    df = df[df.index < end_cutoff]

    _warn_gaps(df, config)
    logger.info(
        "Téléchargement terminé : %d bougies %s %s", len(df), config.symbol, config.timeframe
    )
    return df


def _rows_to_dataframe(rows: list[Any]) -> pd.DataFrame:
    """Convertit les lignes brutes ccxt en DataFrame propre."""
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.astype("float64")


def _warn_gaps(df: pd.DataFrame, config: DataConfig) -> None:
    """Émet un avertissement pour les trous > max_gap_bars bougies consécutives."""
    tf_ms = _TIMEFRAME_MS[config.timeframe]
    expected_delta = pd.Timedelta(milliseconds=tf_ms)
    diffs = df.index.to_series().diff().dropna()
    gap_threshold = expected_delta * (config.max_gap_bars + 1)
    gaps = diffs[diffs > gap_threshold]
    if not gaps.empty:
        for ts, gap in gaps.items():
            n_missing = int(gap / expected_delta) - 1
            logger.warning(
                "Trou de %d bougies manquantes à %s (%s %s)",
                n_missing,
                ts,
                config.symbol,
                config.timeframe,
            )
