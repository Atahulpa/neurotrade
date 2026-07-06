"""Z-score de momentum.

Mesure combien d'écarts-types le prix s'est écarté de sa moyenne mobile.
Normalise le signal de momentum pour le rendre stationnaire.
"""

from __future__ import annotations

import pandas as pd


def momentum_zscore(close: pd.Series, window: int) -> pd.Series:
    """Z-score du prix sur une fenêtre glissante.

    zscore(t) = (close[t] - mean(close, window)) / std(close, window)

    Args:
        close: Prix de clôture.
        window: Taille de la fenêtre glissante en bougies.

    Returns:
        Série de z-scores, même index que close.
        Les premières `window` lignes contiennent NaN.
    """
    mu = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    return (close - mu) / sigma
