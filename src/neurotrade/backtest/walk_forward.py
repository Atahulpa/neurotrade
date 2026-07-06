"""Moteur walk-forward glissant avec purge et embargo.

Principe :
    |── train ──|── purge gap ──|── val ──|── embargo gap ──|── test ──|
                                 ↑ anti-leakage label         ↑ délai exécution

Le purge retire les dernières `purge_bars` bougies du train (leurs labels
ont un horizon qui chevauche le début de val → fuite de label sans purge).
L'embargo retire les premières `embargo_bars` bougies de test (délai
d'exécution simulé — on ne peut pas trader la première bougie post-signal).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from neurotrade.config.schema import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Une fenêtre walk-forward avec indices de train, val et test.

    Invariants garantis par WalkForwardEngine.split() :
    - train_idx.max() < val_idx.min()   (purge gap entre train et val)
    - val_idx.max() < test_idx.min()    (embargo gap entre val et test)
    - Aucun overlap entre train, val, test.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    window_number: int


class WalkForwardEngine:
    """Génère les fenêtres walk-forward glissantes avec purge et embargo.

    Paramètres clés (depuis BacktestConfig) :
    - train_window_bars : taille totale de la fenêtre train+val
    - val_ratio         : fraction de train_window_bars réservée à la validation
    - step_bars         : pas de glissement entre deux fenêtres
    - purge_bars        : bougies à exclure en fin de train (anti-leakage label)
    - embargo_bars      : bougies à exclure en début de test (anti-leakage exécution)
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def split(self, n_samples: int) -> Iterator[WalkForwardWindow]:
        """Génère les fenêtres walk-forward sur `n_samples` observations.

        Args:
            n_samples: Nombre total de barres disponibles.

        Yields:
            WalkForwardWindow avec train/val/test sans fuite temporelle.

        Raises:
            ValueError: Si la configuration produit des fenêtres dégénérées.
        """
        cfg = self.config

        # Taille brute du train avant purge
        train_total = int(cfg.train_window_bars * (1 - cfg.val_ratio))

        # Taille effective du train après purge
        train_size = train_total - cfg.purge_bars
        if train_size <= 0:
            raise ValueError(
                f"purge_bars ({cfg.purge_bars}) >= train_total ({train_total}). "
                "Réduire purge_bars ou augmenter train_window_bars."
            )

        val_size = cfg.train_window_bars - train_total
        if val_size <= 0:
            raise ValueError(
                f"val_size calculée = {val_size} <= 0. "
                "Vérifier val_ratio et train_window_bars."
            )

        logger.debug(
            "WalkForward : train=%d purge_gap=%d val=%d embargo=%d step=%d",
            train_size, cfg.purge_bars, val_size, cfg.embargo_bars, cfg.step_bars,
        )

        window_num = 0
        w_start = 0

        while True:
            train_end = w_start + train_size            # exclusif — [w_start, train_end)
            val_start = w_start + train_total           # saute le purge gap
            val_end = w_start + cfg.train_window_bars   # exclusif — [val_start, val_end)
            test_start = val_end + cfg.embargo_bars     # saute l'embargo
            test_end = val_end + cfg.step_bars          # exclusif — [test_start, test_end)

            if test_end > n_samples:
                break

            train_idx = np.arange(w_start, train_end)
            val_idx = np.arange(val_start, val_end)
            test_idx = np.arange(test_start, test_end)

            logger.debug(
                "Fenêtre %d : train=[%d,%d) val=[%d,%d) test=[%d,%d)",
                window_num, w_start, train_end, val_start, val_end, test_start, test_end,
            )

            yield WalkForwardWindow(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                window_number=window_num,
            )

            w_start += cfg.step_bars
            window_num += 1

    def n_windows(self, n_samples: int) -> int:
        """Compte le nombre de fenêtres sans les stocker en mémoire."""
        return sum(1 for _ in self.split(n_samples))
