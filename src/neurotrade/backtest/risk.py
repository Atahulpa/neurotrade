"""Gestion du risque : sizing, stop journalier, kill-switch.

Sizing : fraction fixe du capital plafonnée à max_position_pct.
Stop journalier : si drawdown du jour > daily_stop_pct → flat + pause 24h.
Kill-switch volatilité : z-score(vol) > vol_zscore_threshold → flat + pause.
Kill-switch spread    : z-score(spread) > spread_zscore_threshold → flat + pause.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from neurotrade.config.schema import RiskConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """Contrôle le sizing des positions et les conditions de kill-switch.

    Utilisation :
        rm = RiskManager(config, initial_capital=10_000)
        size = rm.compute_position_size(signal=1, price=50_000)
        triggered = rm.check_kill_switch(current_vol, vol_mean, vol_std, ...)
        rm.update_daily_pnl(pnl_bar)
        rm.reset_daily() # en début de journée
    """

    def __init__(self, config: RiskConfig, initial_capital: float = 1.0) -> None:
        self.config = config
        self.capital = initial_capital
        self._daily_pnl: float = 0.0
        self._paused: bool = False          # True si stop déclenché ce jour
        self._pause_until: pd.Timestamp | None = None

    # ── Sizing ────────────────────────────────────────────────────────────────

    def compute_position_size(self, signal: int, price: float) -> float:
        """Taille de position en unités de base (ex. BTC), bornée par max_position_pct.

        Args:
            signal: Direction {-1, 0, +1}. 0 → taille nulle.
            price: Prix courant de l'actif (pour convertir en unités).

        Returns:
            Taille positive en unités de base. 0.0 si signal=0 ou système en pause.
        """
        if signal == 0 or self._paused or price <= 0:
            return 0.0
        notional = self.capital * self.config.max_position_pct
        return notional / price

    # ── Stop journalier ───────────────────────────────────────────────────────

    def update_daily_pnl(self, bar_pnl: float) -> bool:
        """Accumule le PnL de la barre courante et déclenche le stop si nécessaire.

        Args:
            bar_pnl: PnL réalisé sur cette barre (peut être positif ou négatif).

        Returns:
            True si le stop journalier vient d'être déclenché sur cette barre.
        """
        if self._paused:
            return False
        self._daily_pnl += bar_pnl
        threshold = -abs(self.config.daily_stop_pct * self.capital)
        if self._daily_pnl <= threshold:
            self._paused = True
            logger.warning(
                "Stop journalier déclenché : PnL_jour=%.4f, seuil=%.4f.",
                self._daily_pnl, threshold,
            )
            return True
        return False

    def reset_daily(self) -> None:
        """Réinitialise le compteur journalier (à appeler en début de journée)."""
        self._daily_pnl = 0.0
        self._paused = False

    # ── Kill-switch volatilité / spread ──────────────────────────────────────

    def check_kill_switch(
        self,
        current_vol: float,
        vol_mean: float,
        vol_std: float,
        spread: float,
        spread_mean: float,
        spread_std: float,
    ) -> bool:
        """Retourne True si le kill-switch doit être déclenché et met le système en pause.

        Le kill-switch se déclenche si :
        - z-score de la volatilité courante > vol_zscore_threshold, OU
        - z-score du spread courant > spread_zscore_threshold.

        Args:
            current_vol: Volatilité réalisée courante (même unité que vol_mean/vol_std).
            vol_mean: Moyenne historique de la volatilité (fenêtre glissante).
            vol_std: Écart-type historique de la volatilité.
            spread: Bid-ask spread courant (ou proxy).
            spread_mean: Moyenne historique du spread.
            spread_std: Écart-type historique du spread.

        Returns:
            True si le kill-switch est déclenché.
        """
        cfg = self.config

        vol_z = _zscore(current_vol, vol_mean, vol_std)
        spread_z = _zscore(spread, spread_mean, spread_std)

        triggered = vol_z > cfg.vol_zscore_threshold or spread_z > cfg.spread_zscore_threshold

        if triggered and not self._paused:
            self._paused = True
            logger.warning(
                "Kill-switch : vol_z=%.2f (seuil=%.1f), spread_z=%.2f (seuil=%.1f).",
                vol_z, cfg.vol_zscore_threshold,
                spread_z, cfg.spread_zscore_threshold,
            )

        return triggered

    @property
    def is_paused(self) -> bool:
        """True si le système est en pause (stop ou kill-switch)."""
        return self._paused

    def resume(self) -> None:
        """Reprend le trading (à appeler manuellement ou en début de session)."""
        self._paused = False

    # ── Sizing walk-forward : calcule les sizes pour une série de signaux ─────

    def compute_sizes_series(
        self,
        signals: pd.Series,
        prices: pd.Series,
    ) -> pd.Series:
        """Calcule la série de tailles de position pour un batch de signaux.

        Applique la règle de sizing barre par barre sans tenir compte du kill-switch
        (le kill-switch est vérifié séparément pendant l'exécution en live).
        Utilisé principalement dans le backtest walk-forward.

        Args:
            signals: Série de signaux {-1, 0, +1}.
            prices: Série de prix (close ou open) alignée sur signals.

        Returns:
            Série de tailles de position (en unités de base).
        """
        size_vals = np.array(
            [self.compute_position_size(int(sig), float(prices.iloc[i]))
             for i, sig in enumerate(signals)]
        )
        return pd.Series(size_vals, index=signals.index)


def _zscore(value: float, mean: float, std: float) -> float:
    """Z-score protégé contre la division par zéro."""
    if std <= 0:
        return 0.0
    return (value - mean) / std


def compute_position_sizes_vectorized(
    signals: np.ndarray,
    prices: np.ndarray,
    capital: float,
    max_position_pct: float,
) -> np.ndarray:
    """Version vectorisée du sizing, sans état (pour tests et batch processing).

    Args:
        signals: Tableau {-1, 0, +1} de forme (n,).
        prices: Tableau de prix de forme (n,).
        capital: Capital total disponible.
        max_position_pct: Fraction max du capital par trade.

    Returns:
        Tableau de tailles de position (en unités de base), forme (n,).
    """
    flat_mask = (signals == 0) | (prices <= 0)
    notional = capital * max_position_pct
    sizes = np.where(flat_mask, 0.0, notional / np.where(prices > 0, prices, 1.0))
    return sizes.astype(np.float64)
