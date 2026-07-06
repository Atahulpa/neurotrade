"""Simulateur d'exécution réaliste : frais, slippage, funding, latence.

Modèle d'exécution :
- Signal au bar t → exécution à l'open du bar t + latency_bars
- Slippage : direction × slippage_bps/10000 × prix
  - Long entry  : on paye plus cher  → entry_price × (1 + slip)
  - Long exit   : on vend moins cher → exit_price  × (1 - slip)
- Frais taker sur les deux jambes (entry + exit)
- Funding rate (perpetual swap) : prélevé prorata temporis par rapport à 8h
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from neurotrade.config.schema import ExecutionConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Représente un trade individuel avec tous ses coûts."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int           # +1 long, -1 short
    entry_price: float       # prix réel d'exécution (après slippage)
    exit_price: float        # prix réel de clôture  (après slippage)
    size: float              # taille en unités de base (ex. BTC)
    fee_paid: float          # frais totaux (entrée + sortie)
    slippage_paid: float     # coût brut du slippage (abs)
    funding_paid: float      # cumul du funding rate sur la durée
    pnl_gross: float         # (exit_raw - entry_raw) × size × direction
    pnl_net: float           # pnl_gross − fees − slippage − funding


def _bars_per_8h(index: pd.DatetimeIndex) -> float:
    """Déduit le nombre de barres dans 8 heures à partir de la fréquence de l'index."""
    if len(index) < 2:
        return 8.0
    ts = pd.Series(index.view("int64"))
    deltas = ts.diff().dropna()
    if deltas.empty:
        return 8.0
    median_ns: float = float(deltas.median())
    if median_ns <= 0:
        return 8.0
    seconds_per_bar = median_ns / 1e9
    return 8 * 3600 / seconds_per_bar


class ExecutionSimulator:
    """Simule l'exécution des ordres à partir de signaux discrets {-1, 0, +1}.

    Utilisation :
        sim = ExecutionSimulator(config)
        trades = sim.simulate(signals, ohlcv, position_sizes)
    """

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    def simulate(
        self,
        signals: pd.Series,
        ohlcv: pd.DataFrame,
        position_sizes: pd.Series,
    ) -> list[Trade]:
        """Simule les trades et retourne la liste des Trade avec coûts complets.

        Args:
            signals: Série {-1, 0, +1} indexée par datetime, un signal par bougie.
            ohlcv: DataFrame OHLCV aligné sur le même index (colonnes : open, close, …).
            position_sizes: Taille de position désirée (en unités de base) par signal.

        Returns:
            Liste de Trade, triée par entry_time croissant.
        """
        cfg = self.config
        slip_frac = cfg.slippage_bps / 10_000

        # Alignment strict sur l'intersection des index
        common_idx = signals.index.intersection(ohlcv.index)
        sig = signals.reindex(common_idx).fillna(0).astype(int)
        bars_df = ohlcv.reindex(common_idx)
        sizes = position_sizes.reindex(common_idx).fillna(0.0)

        n_bars = len(common_idx)
        dti = pd.DatetimeIndex(common_idx)
        bph = _bars_per_8h(dti)

        trades: list[Trade] = []
        latency = cfg.latency_bars

        # État de la position ouverte
        pos_dir = 0
        entry_bar_i = 0
        entry_price_raw = 0.0
        entry_price_slip = 0.0
        entry_size = 0.0

        for i in range(latency, n_bars):
            sig_bar = i - latency
            exec_sig = int(sig.iloc[sig_bar])
            exec_size = float(sizes.iloc[sig_bar])
            bar_open = float(bars_df["open"].iloc[i])

            # Clôture si changement de direction ou retour à plat
            if pos_dir != 0 and exec_sig != pos_dir:
                exit_raw = bar_open
                exit_slip = exit_raw * (1.0 - pos_dir * slip_frac)
                holding = i - entry_bar_i
                funding = (
                    cfg.funding_rate_8h
                    * (holding / bph)
                    * entry_price_slip
                    * entry_size
                )
                fee = cfg.fee_taker * (entry_price_slip + exit_slip) * entry_size
                slippage_cost = slip_frac * (entry_price_raw + exit_raw) * entry_size
                pnl_gross = pos_dir * (exit_raw - entry_price_raw) * entry_size
                pnl_net = (
                    pos_dir * (exit_slip - entry_price_slip) * entry_size
                    - fee
                    - funding
                )

                trades.append(
                    Trade(
                        entry_time=dti[entry_bar_i],
                        exit_time=dti[i],
                        direction=pos_dir,
                        entry_price=entry_price_slip,
                        exit_price=exit_slip,
                        size=entry_size,
                        fee_paid=fee,
                        slippage_paid=slippage_cost,
                        funding_paid=funding,
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                    )
                )
                pos_dir = 0

            # Ouverture si signal non nul et position plate
            if pos_dir == 0 and exec_sig != 0 and exec_size > 0:
                entry_price_raw = bar_open
                entry_price_slip = bar_open * (1.0 + exec_sig * slip_frac)
                pos_dir = exec_sig
                entry_bar_i = i
                entry_size = exec_size

        # Force-close de la position restante en fin de backtest
        if pos_dir != 0 and n_bars > 0:
            i = n_bars - 1
            exit_raw = float(bars_df["open"].iloc[i])
            exit_slip = exit_raw * (1.0 - pos_dir * slip_frac)
            holding = i - entry_bar_i
            funding = (
                cfg.funding_rate_8h * (holding / bph) * entry_price_slip * entry_size
            )
            fee = cfg.fee_taker * (entry_price_slip + exit_slip) * entry_size
            slippage_cost = slip_frac * (entry_price_raw + exit_raw) * entry_size
            pnl_gross = pos_dir * (exit_raw - entry_price_raw) * entry_size
            pnl_net = (
                pos_dir * (exit_slip - entry_price_slip) * entry_size - fee - funding
            )
            trades.append(
                Trade(
                    entry_time=dti[entry_bar_i],
                    exit_time=dti[i],
                    direction=pos_dir,
                    entry_price=entry_price_slip,
                    exit_price=exit_slip,
                    size=entry_size,
                    fee_paid=fee,
                    slippage_paid=slippage_cost,
                    funding_paid=funding,
                    pnl_gross=pnl_gross,
                    pnl_net=pnl_net,
                )
            )

        logger.info("ExecutionSimulator : %d trades générés.", len(trades))
        return trades
