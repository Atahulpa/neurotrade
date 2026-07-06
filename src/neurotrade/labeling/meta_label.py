"""Meta-labeling (optionnel) — filtre les prédictions du modèle primaire.

Implémenté si pertinent après l'étape 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_meta_labeling(
    primary_labels: pd.Series,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> pd.Series:
    """Filtre les labels primaires selon la confiance du modèle secondaire.

    Raises:
        NotImplementedError: Optionnel — implémenté après l'étape 4 si pertinent.
    """
    raise NotImplementedError("apply_meta_labeling est optionnel (étape 4+)")
