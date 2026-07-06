"""Tests pour backtest/ — walk-forward, execution, risk.

DoD étape 5 (walk_forward) :
- Les fenêtres générées ne se chevauchent pas (train/val/test)
- Le purge gap est respecté : train_idx.max() < val_idx.min()
- L'embargo est respecté : val_idx.max() < test_idx.min()
- Les fenêtres sont contiguës (test_n.max() < test_{n+1}.min())
- ValueError si purge_bars >= train_total
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from neurotrade.backtest.walk_forward import WalkForwardEngine
from neurotrade.config.schema import BacktestConfig


def _cfg(**kwargs: object) -> BacktestConfig:
    """Crée un BacktestConfig compact pour les tests."""
    defaults = {
        "train_window_bars": 500,
        "step_bars": 100,
        "val_ratio": 0.2,
        "purge_bars": 10,
        "embargo_bars": 5,
    }
    defaults.update(kwargs)
    return BacktestConfig(**defaults)  # type: ignore[arg-type]


class TestWalkForwardFenêtres:
    def test_au_moins_une_fenetre(self) -> None:
        cfg = _cfg()
        engine = WalkForwardEngine(cfg)
        windows = list(engine.split(2000))
        assert len(windows) > 0

    def test_nombre_fenetres_coherent(self) -> None:
        cfg = _cfg(train_window_bars=500, step_bars=100)
        engine = WalkForwardEngine(cfg)
        windows = list(engine.split(2000))
        assert engine.n_windows(2000) == len(windows)

    def test_window_number_incrementel(self) -> None:
        cfg = _cfg()
        windows = list(WalkForwardEngine(cfg).split(2000))
        for i, w in enumerate(windows):
            assert w.window_number == i

    def test_test_indices_strictement_croissants_entre_fenetres(self) -> None:
        cfg = _cfg()
        windows = list(WalkForwardEngine(cfg).split(2000))
        for w1, w2 in itertools.pairwise(windows):
            assert w1.test_idx.max() < w2.test_idx.min()

    def test_pas_de_fuite_si_n_samples_trop_petit(self) -> None:
        cfg = _cfg()
        windows = list(WalkForwardEngine(cfg).split(300))
        assert len(windows) == 0


class TestAntiLeakagePurge:
    """Purge : train_idx.max() < val_idx.min() — gap garanti."""

    def test_purge_gap_presente(self) -> None:
        cfg = _cfg(purge_bars=20)
        for w in WalkForwardEngine(cfg).split(3000):
            assert w.train_idx.max() < w.val_idx.min(), (
                f"Fenêtre {w.window_number} : train/val se touchent sans purge gap."
            )

    def test_purge_gap_taille_exacte(self) -> None:
        """Le gap entre train_end et val_start doit être exactement purge_bars."""
        purge = 15
        cfg = _cfg(purge_bars=purge)
        for w in WalkForwardEngine(cfg).split(3000):
            gap = int(w.val_idx.min()) - int(w.train_idx.max()) - 1
            assert gap == purge, (
                f"Fenêtre {w.window_number} : gap={gap}, attendu={purge}."
            )

    def test_pas_overlap_train_val(self) -> None:
        cfg = _cfg()
        for w in WalkForwardEngine(cfg).split(3000):
            assert len(np.intersect1d(w.train_idx, w.val_idx)) == 0

    def test_pas_overlap_train_test(self) -> None:
        cfg = _cfg()
        for w in WalkForwardEngine(cfg).split(3000):
            assert len(np.intersect1d(w.train_idx, w.test_idx)) == 0

    def test_pas_overlap_val_test(self) -> None:
        cfg = _cfg()
        for w in WalkForwardEngine(cfg).split(3000):
            assert len(np.intersect1d(w.val_idx, w.test_idx)) == 0


class TestAntiLeakageEmbargo:
    """Embargo : val_idx.max() < test_idx.min() — gap garanti."""

    def test_embargo_gap_presente(self) -> None:
        cfg = _cfg(embargo_bars=10)
        for w in WalkForwardEngine(cfg).split(3000):
            assert w.val_idx.max() < w.test_idx.min(), (
                f"Fenêtre {w.window_number} : val/test sans embargo gap."
            )

    def test_embargo_gap_taille_exacte(self) -> None:
        embargo = 8
        cfg = _cfg(embargo_bars=embargo)
        for w in WalkForwardEngine(cfg).split(3000):
            gap = int(w.test_idx.min()) - int(w.val_idx.max()) - 1
            assert gap == embargo, (
                f"Fenêtre {w.window_number} : gap={gap}, attendu={embargo}."
            )

    def test_embargo_zero_val_test_contigus(self) -> None:
        """Avec embargo_bars=0, test commence immédiatement après val."""
        cfg = _cfg(embargo_bars=0)
        for w in WalkForwardEngine(cfg).split(3000):
            assert int(w.test_idx.min()) == int(w.val_idx.max()) + 1


class TestErreurs:
    def test_purge_trop_grand_leve_valueerror(self) -> None:
        cfg = _cfg(train_window_bars=500, val_ratio=0.2, purge_bars=400)
        with pytest.raises(ValueError, match="purge_bars"):
            list(WalkForwardEngine(cfg).split(5000))

    def test_indices_dans_bornes(self) -> None:
        n = 3000
        cfg = _cfg()
        for w in WalkForwardEngine(cfg).split(n):
            assert w.train_idx.min() >= 0
            assert w.test_idx.max() <= n


# ─── Étape 6 : ExecutionSimulator ────────────────────────────────────────────

import pandas as pd  # noqa: E402

from neurotrade.backtest.execution import ExecutionSimulator  # noqa: E402
from neurotrade.config.schema import ExecutionConfig  # noqa: E402


def _exec_cfg(**kwargs: object) -> ExecutionConfig:
    defaults: dict[str, object] = {
        "fee_maker": 0.0002,
        "fee_taker": 0.0004,
        "slippage_bps": 1.0,
        "funding_rate_8h": 0.0001,
        "latency_bars": 1,
    }
    defaults.update(kwargs)
    return ExecutionConfig(**defaults)  # type: ignore[arg-type]


def _make_ohlcv(prices: list[float]) -> pd.DataFrame:
    """Crée un DataFrame OHLCV minimal avec open=close=price pour chaque barre."""
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="1h")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0},
        index=idx,
    )


class TestExecutionSimulatorPnLManuel:
    """Vérification analytique du PnL sur des scénarios simples.

    On choisit slippage_bps=0 et fee_taker=0 pour isoler le PnL brut,
    puis on teste les coûts séparément.
    """

    def test_long_parfait_sans_frais(self) -> None:
        """Long entre 100 et 110 : PnL = 10 par unité."""
        cfg = _exec_cfg(fee_taker=0.0, slippage_bps=0.0, funding_rate_8h=0.0, latency_bars=1)
        # signal bar0→exec bar1 @ 100 ; signal bar2→exec bar3 @ 110
        prices = [100.0, 100.0, 110.0, 110.0]
        ohlcv = _make_ohlcv(prices)
        signals = pd.Series([1, 1, 0, 0], index=ohlcv.index, dtype=int)
        sizes = pd.Series([1.0, 1.0, 1.0, 1.0], index=ohlcv.index)

        sim = ExecutionSimulator(cfg)
        trades = sim.simulate(signals, ohlcv, sizes)

        assert len(trades) == 1
        t = trades[0]
        assert t.direction == 1
        assert abs(t.entry_price - 100.0) < 1e-9
        assert abs(t.exit_price - 110.0) < 1e-9
        assert abs(t.pnl_gross - 10.0) < 1e-9
        assert abs(t.pnl_net - 10.0) < 1e-9

    def test_short_parfait_sans_frais(self) -> None:
        """Short entre 110 et 100 : PnL = 10 par unité."""
        cfg = _exec_cfg(fee_taker=0.0, slippage_bps=0.0, funding_rate_8h=0.0, latency_bars=1)
        prices = [110.0, 110.0, 100.0, 100.0]
        ohlcv = _make_ohlcv(prices)
        signals = pd.Series([-1, -1, 0, 0], index=ohlcv.index, dtype=int)
        sizes = pd.Series([1.0] * 4, index=ohlcv.index)

        trades = ExecutionSimulator(cfg).simulate(signals, ohlcv, sizes)

        assert len(trades) == 1
        t = trades[0]
        assert t.direction == -1
        assert abs(t.pnl_gross - 10.0) < 1e-9
        assert abs(t.pnl_net - 10.0) < 1e-9

    def test_frais_reduisent_pnl(self) -> None:
        """Avec fee_taker=0.001, PnL net < PnL brut."""
        cfg = _exec_cfg(fee_taker=0.001, slippage_bps=0.0, funding_rate_8h=0.0)
        prices = [100.0, 100.0, 110.0, 110.0]
        ohlcv = _make_ohlcv(prices)
        signals = pd.Series([1, 1, 0, 0], index=ohlcv.index, dtype=int)
        sizes = pd.Series([1.0] * 4, index=ohlcv.index)

        trades = ExecutionSimulator(cfg).simulate(signals, ohlcv, sizes)
        t = trades[0]
        expected_fee = 0.001 * (100.0 + 110.0) * 1.0
        assert abs(t.fee_paid - expected_fee) < 1e-6
        assert t.pnl_net < t.pnl_gross

    def test_slippage_reduit_pnl(self) -> None:
        """Avec slippage, les prix d'exécution sont défavorables."""
        cfg = _exec_cfg(fee_taker=0.0, slippage_bps=10.0, funding_rate_8h=0.0)
        prices = [100.0, 100.0, 110.0, 110.0]
        ohlcv = _make_ohlcv(prices)
        signals = pd.Series([1, 1, 0, 0], index=ohlcv.index, dtype=int)
        sizes = pd.Series([1.0] * 4, index=ohlcv.index)

        trades = ExecutionSimulator(cfg).simulate(signals, ohlcv, sizes)
        t = trades[0]
        slip_frac = 10.0 / 10_000
        # Long entry : achète plus cher, exit : vend moins cher
        expected_entry = 100.0 * (1 + slip_frac)
        expected_exit = 110.0 * (1 - slip_frac)
        assert abs(t.entry_price - expected_entry) < 1e-6
        assert abs(t.exit_price - expected_exit) < 1e-6
        assert t.pnl_net < t.pnl_gross

    def test_latence_retarde_execution(self) -> None:
        """Latency=2 : le signal au bar 0 est exécuté à l'open du bar 2."""
        cfg = _exec_cfg(fee_taker=0.0, slippage_bps=0.0, funding_rate_8h=0.0, latency_bars=2)
        prices = [50.0, 75.0, 100.0, 100.0, 80.0, 80.0]
        ohlcv = _make_ohlcv(prices)
        signals = pd.Series([1, 1, 1, 1, 0, 0], index=ohlcv.index, dtype=int)
        sizes = pd.Series([1.0] * 6, index=ohlcv.index)

        trades = ExecutionSimulator(cfg).simulate(signals, ohlcv, sizes)
        # Entrée au bar 2 (open=100), sortie au bar 6 (force-close, open=80)
        assert len(trades) >= 1
        t = trades[0]
        assert abs(t.entry_price - 100.0) < 1e-9  # bar 2

    def test_pas_de_trade_si_size_zero(self) -> None:
        """Aucun trade si position_sizes=0."""
        cfg = _exec_cfg()
        prices = [100.0, 100.0, 110.0, 110.0]
        ohlcv = _make_ohlcv(prices)
        signals = pd.Series([1, 1, 0, 0], index=ohlcv.index, dtype=int)
        sizes = pd.Series([0.0] * 4, index=ohlcv.index)

        trades = ExecutionSimulator(cfg).simulate(signals, ohlcv, sizes)
        assert len(trades) == 0

    def test_reversal_genere_deux_trades(self) -> None:
        """Passage direct de +1 à -1 génère close long + open short = 2 trades."""
        cfg = _exec_cfg(fee_taker=0.0, slippage_bps=0.0, funding_rate_8h=0.0)
        prices = [100.0] * 6
        ohlcv = _make_ohlcv(prices)
        # Long 2 bars, puis reversal short 2 bars
        signals = pd.Series([1, 1, -1, -1, 0, 0], index=ohlcv.index, dtype=int)
        sizes = pd.Series([1.0] * 6, index=ohlcv.index)

        trades = ExecutionSimulator(cfg).simulate(signals, ohlcv, sizes)
        assert len(trades) == 2
        assert trades[0].direction == 1
        assert trades[1].direction == -1


# ─── Étape 7 : RiskManager ───────────────────────────────────────────────────

from neurotrade.backtest.risk import RiskManager, compute_position_sizes_vectorized  # noqa: E402
from neurotrade.config.schema import RiskConfig  # noqa: E402


def _risk_cfg(**kwargs: object) -> RiskConfig:
    defaults: dict[str, object] = {
        "max_position_pct": 0.02,
        "daily_stop_pct": 0.01,
        "vol_zscore_threshold": 3.0,
        "spread_zscore_threshold": 4.0,
    }
    defaults.update(kwargs)
    return RiskConfig(**defaults)  # type: ignore[arg-type]


class TestSizing:
    def test_taille_zero_si_signal_nul(self) -> None:
        rm = RiskManager(_risk_cfg(), initial_capital=10_000)
        assert rm.compute_position_size(0, price=50_000) == 0.0

    def test_taille_correcte_long(self) -> None:
        capital = 10_000
        max_pct = 0.02
        price = 50_000.0
        rm = RiskManager(_risk_cfg(max_position_pct=max_pct), initial_capital=capital)
        size = rm.compute_position_size(1, price=price)
        expected = (capital * max_pct) / price
        assert abs(size - expected) < 1e-9

    def test_taille_short_egale_long(self) -> None:
        rm = RiskManager(_risk_cfg(), initial_capital=10_000)
        size_long = rm.compute_position_size(1, price=40_000)
        size_short = rm.compute_position_size(-1, price=40_000)
        assert abs(size_long - size_short) < 1e-9

    def test_taille_zero_si_system_en_pause(self) -> None:
        rm = RiskManager(_risk_cfg(), initial_capital=10_000)
        rm._paused = True
        assert rm.compute_position_size(1, price=50_000) == 0.0

    def test_vectorized_sizing_coherent(self) -> None:
        import numpy as np
        signals = np.array([1, 0, -1, 1])
        prices = np.array([100.0, 100.0, 100.0, 50.0])
        sizes = compute_position_sizes_vectorized(
            signals, prices, capital=1000, max_position_pct=0.02
        )
        assert sizes[1] == 0.0          # signal=0 → taille nulle
        assert abs(sizes[0] - sizes[2]) < 1e-9   # même prix → mêmes tailles


class TestStopJournalier:
    def test_pnl_positif_ne_declenche_pas_stop(self) -> None:
        rm = RiskManager(_risk_cfg(daily_stop_pct=0.01), initial_capital=1000)
        triggered = rm.update_daily_pnl(5.0)
        assert not triggered
        assert not rm.is_paused

    def test_pnl_negatif_declenche_stop(self) -> None:
        rm = RiskManager(_risk_cfg(daily_stop_pct=0.01), initial_capital=1000)
        # Seuil = 1% × 1000 = 10 → perte de 11 déclenche le stop
        triggered = rm.update_daily_pnl(-11.0)
        assert triggered
        assert rm.is_paused

    def test_reset_daily_reprend_trading(self) -> None:
        rm = RiskManager(_risk_cfg(daily_stop_pct=0.01), initial_capital=1000)
        rm.update_daily_pnl(-100.0)
        assert rm.is_paused
        rm.reset_daily()
        assert not rm.is_paused

    def test_pas_de_double_stop(self) -> None:
        """Une fois en pause, update_daily_pnl ne déclenche plus rien."""
        rm = RiskManager(_risk_cfg(daily_stop_pct=0.01), initial_capital=1000)
        rm.update_daily_pnl(-100.0)
        triggered2 = rm.update_daily_pnl(-100.0)
        assert not triggered2


class TestKillSwitch:
    def test_vol_zscore_eleve_declenche(self) -> None:
        rm = RiskManager(_risk_cfg(vol_zscore_threshold=3.0))
        triggered = rm.check_kill_switch(
            current_vol=4.0, vol_mean=1.0, vol_std=1.0,  # z = (4-1)/1 = 3 ← seuil exact
            spread=0.0, spread_mean=0.0, spread_std=1.0,
        )
        # z=3.0 n'est PAS > 3.0 (strict), donc pas déclenché
        assert not triggered

    def test_vol_zscore_depasse_declenche(self) -> None:
        rm = RiskManager(_risk_cfg(vol_zscore_threshold=3.0))
        triggered = rm.check_kill_switch(
            current_vol=4.1, vol_mean=1.0, vol_std=1.0,  # z = 3.1 > 3.0
            spread=0.0, spread_mean=0.0, spread_std=1.0,
        )
        assert triggered
        assert rm.is_paused

    def test_spread_zscore_depasse_declenche(self) -> None:
        rm = RiskManager(_risk_cfg(spread_zscore_threshold=4.0))
        triggered = rm.check_kill_switch(
            current_vol=0.0, vol_mean=0.0, vol_std=1.0,
            spread=5.0, spread_mean=1.0, spread_std=1.0,  # z = 4.0 → pas strict
        )
        assert not triggered   # z=4.0 n'est pas > 4.0

    def test_spread_zscore_strict_depasse(self) -> None:
        rm = RiskManager(_risk_cfg(spread_zscore_threshold=4.0))
        triggered = rm.check_kill_switch(
            current_vol=0.0, vol_mean=0.0, vol_std=1.0,
            spread=5.1, spread_mean=1.0, spread_std=1.0,  # z = 4.1 > 4.0
        )
        assert triggered

    def test_std_zero_ne_crase_pas(self) -> None:
        """std=0 → z-score retourne 0 → pas de kill-switch."""
        rm = RiskManager(_risk_cfg())
        triggered = rm.check_kill_switch(
            current_vol=999.0, vol_mean=0.0, vol_std=0.0,
            spread=999.0, spread_mean=0.0, spread_std=0.0,
        )
        assert not triggered
