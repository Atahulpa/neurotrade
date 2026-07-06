"""Features basées sur le volume.

Deux features simples :
- volume relatif : volume / mean_volume(window) — capte les pics d'activité
- log-volume : log(volume + 1) — normalise la distribution très skewed du volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volume_features(volume: pd.Series, window: int = 20) -> pd.DataFrame:
    """Calcule les features de volume.

    Args:
        volume: Série de volumes.
        window: Fenêtre pour le volume moyen relatif.

    Returns:
        DataFrame avec colonnes [vol_rel, log_vol].
    """
    mean_vol = volume.rolling(window).mean()
    vol_rel = volume / mean_vol.clip(lower=1e-8)  # clip évite division par zéro
    log_vol = np.log1p(volume)
    return pd.DataFrame({"vol_rel": vol_rel, "log_vol": log_vol}, index=volume.index)
