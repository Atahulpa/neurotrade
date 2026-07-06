"""Log-returns multi-horizons.

log_return(t, h) = log(close[t] / close[t-h])

Ce module est un cœur pur : zéro I/O, zéro réseau, zéro torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Calcule les log-returns pour chaque horizon.

    Args:
        close: Série de prix de clôture, index DatetimeIndex.
        horizons: Liste d'horizons en nombre de bougies (ex. [1, 5, 15, 60]).

    Returns:
        DataFrame avec une colonne `ret_h` par horizon, même index que close.
        Les premières `max(horizons)` lignes contiennent NaN.
    """
    result: dict[str, pd.Series] = {}
    log_close = np.log(close)
    for h in horizons:
        result[f"ret_{h}"] = log_close - log_close.shift(h)
    return pd.DataFrame(result, index=close.index)
