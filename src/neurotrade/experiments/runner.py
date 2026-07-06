"""Orchestrateur des expériences.

Chaîne : data → features → labels → walk-forward → model → execution → eval → report.

Journalise le nombre de configs testées (n_trials) pour le calcul du DSR.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from neurotrade.backtest.execution import ExecutionSimulator, Trade
from neurotrade.backtest.risk import RiskManager
from neurotrade.backtest.walk_forward import WalkForwardEngine
from neurotrade.config.schema import AppConfig, ModelConfig
from neurotrade.data.cache import load_or_fetch
from neurotrade.data.fetcher import fetch_ohlcv
from neurotrade.evaluation.metrics import BacktestMetrics, compute_metrics
from neurotrade.evaluation.report import generate_report
from neurotrade.features.momentum import momentum_zscore
from neurotrade.features.returns import log_returns
from neurotrade.features.scaler import FeatureScaler
from neurotrade.features.volatility import realized_volatility
from neurotrade.features.volume import volume_features
from neurotrade.labeling.triple_barrier import apply_triple_barrier
from neurotrade.models.base import BaseModel
from neurotrade.models.cnn1d import CNN1D
from neurotrade.models.lstm import LSTMModel
from neurotrade.models.mlp import MLP
from neurotrade.models.trainer import Trainer

logger = logging.getLogger(__name__)

# Barres par an selon le timeframe (pour annualisation des métriques)
_BARS_PER_YEAR: dict[str, int] = {
    "1m": 525_960,
    "5m": 105_192,
    "15m": 35_064,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 252,
}


class ExperimentRunner:
    """Lance un run complet et journalise la config + les métriques.

    Utilisation :
        runner = ExperimentRunner(config)
        metrics = runner.run()
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._n_windows_run = 0

    def run(self) -> BacktestMetrics:
        """Exécute le pipeline complet data→features→labels→model→backtest→eval.

        Returns:
            BacktestMetrics du run complet (OOS uniquement).

        Raises:
            ValueError: Si pas assez de données valides.
        """
        cfg = self.config
        torch.manual_seed(cfg.meta.seed)
        np.random.seed(cfg.meta.seed)

        # ── 1. Données ────────────────────────────────────────────────────────
        df = load_or_fetch(cfg.data, fetch_ohlcv)
        logger.info("Données chargées : %d barres (%s)", len(df), cfg.data.symbol)

        # ── 2. Features + positions valides ──────────────────────────────────
        X_flat, valid_positions = _build_features_aligned(df, cfg)
        n_feat = X_flat.shape[1]
        logger.info("Features : %d x %d (apres NaN drop)", *X_flat.shape)

        # ── 3. Labels ─────────────────────────────────────────────────────────
        labels_all = apply_triple_barrier(df, cfg.labeling)
        # Aligner sur les positions valides et filtrer les NaN finaux
        labels_valid = labels_all.iloc[valid_positions].values  # (n_valid,)
        label_valid_mask = ~np.isnan(labels_valid)

        valid_positions = valid_positions[label_valid_mask]
        X_flat = X_flat[label_valid_mask]
        y_valid = labels_valid[label_valid_mask]
        n_valid = len(valid_positions)

        if n_valid < 200:
            raise ValueError(
                f"Trop peu de barres valides ({n_valid}). "
                "Augmenter la période de données ou réduire les fenêtres."
            )
        logger.info("Barres valides (features + labels) : %d", n_valid)

        # ── 4. Walk-forward ───────────────────────────────────────────────────
        engine = WalkForwardEngine(cfg.backtest)
        all_test_returns: list[float] = []
        all_trades: list[Trade] = []
        window_size = cfg.model.window_size
        w_offset = max(0, window_size - 1)

        for window in engine.split(n_valid):
            self._n_windows_run += 1
            wn = self._n_windows_run

            # ── Scaler anti-fuite (fit sur train uniquement) ──────────────
            train_idx = window.train_idx
            test_idx = window.test_idx

            scaler = FeatureScaler()
            scaler.fit(X_flat[train_idx])
            X_scaled = scaler.transform(X_flat)

            # ── Windowing pour CNN/LSTM ────────────────────────────────────
            X_w, y_w = _window_arrays(X_scaled, y_valid, window_size)

            def _to_w(idx: np.ndarray) -> np.ndarray:
                """Flat index → windowed index, masque les trop courts."""
                m = idx >= w_offset
                return idx[m] - w_offset

            tr_w = _to_w(train_idx)
            val_w = _to_w(window.val_idx)
            te_w = _to_w(test_idx)

            if len(tr_w) < 20 or len(te_w) == 0:
                logger.debug("Fenêtre %d ignorée (tr=%d, te=%d).", wn, len(tr_w), len(te_w))
                continue

            X_train = X_w[tr_w].astype(np.float32)
            y_train = y_w[tr_w]
            X_val = X_w[val_w].astype(np.float32) if len(val_w) > 0 else X_train[:5]
            y_val_arr = y_w[val_w] if len(val_w) > 0 else y_train[:5]
            X_test = X_w[te_w].astype(np.float32)

            # ── Modèle ────────────────────────────────────────────────────
            torch.manual_seed(cfg.meta.seed + wn)
            model = _build_model(cfg.model, n_feat)
            Trainer(model, cfg.model).train(X_train, y_train, X_val, y_val_arr)

            # ── Prédictions ───────────────────────────────────────────────
            preds = model.predict(X_test)  # {-1, 0, +1}

            # ── Mapping vers OHLCV original ───────────────────────────────
            test_flat_idx = te_w + w_offset
            test_orig_idx = valid_positions[test_flat_idx]
            test_times = pd.DatetimeIndex(df.index[test_orig_idx])
            ohlcv_test = df.iloc[test_orig_idx][["open", "high", "low", "close", "volume"]].copy()
            ohlcv_test.index = test_times

            signals = pd.Series(preds.astype(int), index=test_times)
            prices = pd.Series(ohlcv_test["close"].values, index=test_times, dtype=float)

            # ── Risk management → sizes ────────────────────────────────────
            rm = RiskManager(cfg.risk, initial_capital=1.0)
            sizes = rm.compute_sizes_series(signals, prices)

            # ── Exécution simulée ─────────────────────────────────────────
            sim = ExecutionSimulator(cfg.execution)
            trades = sim.simulate(signals, ohlcv_test, sizes)
            all_trades.extend(trades)

            # Bar-level returns (PnL net alloué sur la barre de sortie)
            bar_returns = np.zeros(len(test_times), dtype=float)
            test_times_arr = test_times.to_numpy()
            for trade in trades:
                i_exit = int(np.searchsorted(test_times_arr, np.datetime64(trade.exit_time)))
                if i_exit < len(bar_returns):
                    bar_returns[i_exit] += trade.pnl_net
            all_test_returns.extend(bar_returns.tolist())

            logger.debug(
                "Fenêtre %d : %d trades, Σpnl_net=%.4f",
                wn, len(trades), sum(t.pnl_net for t in trades),
            )

        logger.info(
            "Walk-forward terminé : %d fenêtres, %d trades, %d barres OOS.",
            self._n_windows_run, len(all_trades), len(all_test_returns),
        )

        if not all_test_returns:
            raise ValueError("Aucune barre OOS produite — configuration trop restrictive.")

        # ── 5. Métriques ──────────────────────────────────────────────────────
        bpy = _BARS_PER_YEAR.get(cfg.data.timeframe, 8760)
        pnl_net = sum(t.pnl_net for t in all_trades)
        metrics = compute_metrics(
            returns=np.array(all_test_returns),
            n_trades=len(all_trades),
            pnl_net=pnl_net,
            risk_free_rate=cfg.evaluation.risk_free_rate,
            n_trials=max(1, self._n_windows_run),
            bars_per_year=bpy,
        )

        # ── 6. Rapport ────────────────────────────────────────────────────────
        generate_report(metrics, np.array(all_test_returns), cfg.meta.run_name, cfg.evaluation)

        return metrics


# ── Fonctions helpers (module-level, pur) ────────────────────────────────────

def _build_features_aligned(
    df: pd.DataFrame,
    cfg: AppConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit les features et retourne (X, valid_positions) sans NaN.

    valid_positions : indices dans df.index des lignes valides.
    Réplique la logique de build_features() mais conserve le mapping de positions.
    """
    fc = cfg.features
    parts: list[pd.DataFrame | pd.Series] = []
    parts.append(log_returns(df["close"], fc.horizons))
    parts.append(realized_volatility(df["close"], fc.vol_window).rename("vol"))
    parts.append(momentum_zscore(df["close"], fc.momentum_window).rename("momentum"))
    if fc.use_volume:
        parts.append(volume_features(df["volume"], window=fc.vol_window))

    combined = pd.concat(parts, axis=1)
    valid_mask = ~combined.isna().any(axis=1)
    valid_positions = np.where(valid_mask.to_numpy())[0]
    X = combined.iloc[valid_positions].to_numpy(dtype=np.float64)
    return X, valid_positions


def _window_arrays(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Applique un sliding window sur X pour CNN/LSTM.

    Retourne (X_w, y_w) où X_w[i] = X[i:i+window_size].
    y_w[i] correspond à la barre X[i+window_size-1] (dernier de la fenêtre).
    """
    if window_size <= 1:
        return X, y
    n, n_feat = X.shape
    if n < window_size:
        return np.empty((0, window_size, n_feat), dtype=X.dtype), np.empty(0, dtype=y.dtype)
    X_w = np.lib.stride_tricks.sliding_window_view(X, (window_size, n_feat))
    X_w = X_w.reshape(-1, window_size, n_feat)  # (n - ws + 1, ws, n_feat)
    y_w = y[window_size - 1:]                   # aligne sur le dernier bar de chaque fenêtre
    return X_w, y_w


def _build_model(config: ModelConfig, n_features: int) -> BaseModel:
    """Instancie le bon type de modèle selon config.arch."""
    if config.arch == "mlp":
        return MLP(input_dim=n_features, config=config)
    if config.arch == "cnn1d":
        return CNN1D(n_features=n_features, config=config)
    if config.arch == "lstm":
        return LSTMModel(n_features=n_features, config=config)
    raise ValueError(f"Architecture inconnue : {config.arch!r}")
