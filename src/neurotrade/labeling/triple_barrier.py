"""Labeling triple-barrier (TP / SL / horizon).

Pour chaque bougie t, on regarde lequel des trois événements se produit
en premier dans les horizon_bars bougies suivantes :
  1. Take-profit (TP) : high[t+i] >= close[t] * (1 + tp_pct) → label +1
  2. Stop-loss  (SL)  : low[t+i]  <= close[t] * (1 - sl_pct) → label -1
  3. Horizon expiré   : ni TP ni SL dans horizon_bars bougies  → label  0

Implémentation vectorisée numpy via sliding_window_view :
  - Complexité temporelle O(n × H) mais en numpy pur (pas de boucle Python)
  - Pour n=1M et H=60 : ~56 millions de comparaisons en quelques secondes
  - Mémoire : (n - H) × H × 8 octets ≈ 450 Mo pour ces paramètres

Référence : López de Prado, *Advances in Financial Machine Learning*, chap. 3.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from neurotrade.config.schema import LabelingConfig

logger = logging.getLogger(__name__)


def apply_triple_barrier(
    df: pd.DataFrame,
    config: LabelingConfig,
) -> pd.Series:
    """Applique le labeling triple-barrier sur un DataFrame OHLCV.

    Args:
        df: DataFrame avec colonnes [open, high, low, close, volume],
            index DatetimeIndex.
        config: Paramètres TP/SL/horizon.

    Returns:
        pd.Series de float {-1.0, 0.0, +1.0, NaN}, même index que df.
        Les dernières horizon_bars observations valent NaN (pas labelisable).
    """
    n = len(df)
    H = config.horizon_bars
    n_valid = n - H  # nombre de bars qu'on peut étiqueter

    if n_valid <= 0:
        logger.warning("Pas assez de données pour labeler (n=%d, H=%d).", n, H)
        return pd.Series(np.nan, index=df.index, dtype="float64")

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    # Prix cibles TP et SL pour chaque barre de référence
    tp_prices = close[:n_valid] * (1.0 + config.tp_pct)  # (n_valid,)
    sl_prices = close[:n_valid] * (1.0 - config.sl_pct)  # (n_valid,)

    # Fenêtres glissantes des H bougies futures pour chaque barre i
    # sliding_window_view(arr, H+1)[i] = arr[i .. i+H]
    # → on prend [:, 1:] pour avoir arr[i+1 .. i+H] (futures uniquement)
    high_windows = np.lib.stride_tricks.sliding_window_view(high, H + 1)[:n_valid, 1:]
    low_windows = np.lib.stride_tricks.sliding_window_view(low, H + 1)[:n_valid, 1:]

    # Masques de déclenchement : (n_valid, H)
    tp_hits = high_windows >= tp_prices[:, None]
    sl_hits = low_windows <= sl_prices[:, None]

    # Index (0-based) de la première barre qui déclenche chaque barrière
    # Si jamais déclenché, on retourne H (hors fenêtre → neutre par défaut)
    tp_first = np.where(tp_hits.any(axis=1), tp_hits.argmax(axis=1), H)
    sl_first = np.where(sl_hits.any(axis=1), sl_hits.argmax(axis=1), H)

    # Attribution du label : la première barrière touchée gagne
    # Cas "les deux même barre" : TP gagne (hypothèse conservatrice)
    labels = np.select(
        [
            (tp_first < H) & (tp_first <= sl_first),   # TP premier ou ex-aequo
            (sl_first < H) & (sl_first < tp_first),    # SL premier
        ],
        [1.0, -1.0],
        default=0.0,  # aucun déclenchement → neutre
    )

    # Dernières H barres : pas étiquetables
    result = np.full(n, np.nan)
    result[:n_valid] = labels

    n_long = int((labels == 1.0).sum())
    n_short = int((labels == -1.0).sum())
    n_neutral = int((labels == 0.0).sum())
    logger.info(
        "Labels triple-barrier : +1=%d (%.1f%%), -1=%d (%.1f%%), 0=%d (%.1f%%)",
        n_long, 100 * n_long / n_valid,
        n_short, 100 * n_short / n_valid,
        n_neutral, 100 * n_neutral / n_valid,
    )

    return pd.Series(result, index=df.index, dtype="float64")
