"""Volatilité réalisée.

Estimateur : écart-type glissant des log-returns 1-bougie,
annualisé sur la fenêtre demandée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Calcule la volatilité réalisée sur `window` bougies.

    Vol = std(log_return_1bar, window) × √window
    Le facteur √window annualise la vol sur la fenêtre glissante.

    Args:
        close: Prix de clôture.
        window: Taille de la fenêtre glissante en bougies.

    Returns:
        Série de volatilité, même index que close.
        Les premières `window` lignes contiennent NaN.
    """
    log_ret: pd.Series = np.log(close / close.shift(1))  # type: ignore[assignment]
    result: pd.Series = log_ret.rolling(window).std() * np.sqrt(window)
    return result
