"""Tests pour labeling/ — triple-barrier sur 4 cas connus construits à la main.

DoD étape 3 : les labels sont vérifiés analytiquement avant de faire tourner
le code sur des données réelles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurotrade.config.schema import LabelingConfig
from neurotrade.labeling.triple_barrier import apply_triple_barrier


def _make_df(close: list[float], high: list[float], low: list[float]) -> pd.DataFrame:
    """Construit un DataFrame OHLCV minimal pour les tests."""
    n = len(close)
    dates = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100.0] * n,
        },
        index=dates,
    )


class TestTripleBarrierCasConnus:
    """Quatre cas construits à la main avec résultat analytiquement connu."""

    def _config(self) -> LabelingConfig:
        return LabelingConfig(tp_pct=0.01, sl_pct=0.01, horizon_bars=3)

    # ── Cas 1 : TP atteint à la bougie suivante ────────────────────────────────

    def test_cas1_tp_atteint_immediatement(self) -> None:
        """
        Référence : close[0] = 100, TP = 101.
        Barre 1 : high = 102 → TP touché → label +1.
        """
        df = _make_df(
            close=[100.0, 100.0, 100.0, 100.0, 100.0],
            high=[100.0, 102.0, 100.0, 100.0, 100.0],  # Barre 1 : high > TP
            low=[99.0, 99.0, 99.0, 99.0, 99.0],
        )
        labels = apply_triple_barrier(df, self._config())
        assert labels.iloc[0] == pytest.approx(1.0), "Label 0 devrait être +1 (TP)"

    # ── Cas 2 : SL atteint à la bougie suivante ────────────────────────────────

    def test_cas2_sl_atteint_immediatement(self) -> None:
        """
        Référence : close[0] = 100, SL = 99.
        Barre 1 : low = 98 → SL touché → label -1.
        """
        df = _make_df(
            close=[100.0, 100.0, 100.0, 100.0, 100.0],
            high=[100.0, 100.0, 100.0, 100.0, 100.0],
            low=[100.0, 98.0, 100.0, 100.0, 100.0],  # Barre 1 : low < SL
        )
        labels = apply_triple_barrier(df, self._config())
        assert labels.iloc[0] == pytest.approx(-1.0), "Label 0 devrait être -1 (SL)"

    # ── Cas 3 : Horizon expiré sans TP ni SL ──────────────────────────────────

    def test_cas3_neutre_horizon_expire(self) -> None:
        """
        Aucune barrière touchée dans les 3 prochaines bougies → label 0.
        """
        df = _make_df(
            close=[100.0, 100.0, 100.0, 100.0, 100.0],
            high=[100.0, 100.5, 100.3, 100.2, 100.0],  # Jamais ≥ 101
            low=[100.0, 99.8, 99.9, 99.7, 100.0],      # Jamais ≤ 99
        )
        labels = apply_triple_barrier(df, self._config())
        assert labels.iloc[0] == pytest.approx(0.0), "Label 0 devrait être 0 (neutre)"

    # ── Cas 4 : SL avant TP ───────────────────────────────────────────────────

    def test_cas4_sl_avant_tp(self) -> None:
        """
        Barre 1 : SL touché.
        Barre 2 : TP aurait pu être touché, mais SL déjà déclenché.
        → label -1.
        """
        df = _make_df(
            close=[100.0, 100.0, 100.0, 100.0, 100.0],
            high=[100.0, 100.5, 102.0, 100.0, 100.0],  # Barre 2 : TP, mais trop tard
            low=[100.0, 98.0, 99.0, 100.0, 100.0],     # Barre 1 : SL en premier
        )
        labels = apply_triple_barrier(df, self._config())
        assert labels.iloc[0] == pytest.approx(-1.0), "SL en barre 1 devrait gagner"

    # ── Vérifications structurelles ────────────────────────────────────────────

    def test_dernieres_barres_nan(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """Les horizon_bars dernières bougies doivent valoir NaN."""
        cfg = LabelingConfig(tp_pct=0.002, sl_pct=0.001, horizon_bars=10)
        labels = apply_triple_barrier(synthetic_ohlcv, cfg)
        assert labels.iloc[-10:].isna().all()

    def test_labels_dans_valeurs_valides(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """Les labels valides doivent être dans {-1, 0, +1}."""
        cfg = LabelingConfig(tp_pct=0.002, sl_pct=0.001, horizon_bars=10)
        labels = apply_triple_barrier(synthetic_ohlcv, cfg)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_nombre_total_correct(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """Longueur du résultat = longueur de l'entrée."""
        cfg = LabelingConfig(tp_pct=0.002, sl_pct=0.001, horizon_bars=20)
        labels = apply_triple_barrier(synthetic_ohlcv, cfg)
        assert len(labels) == len(synthetic_ohlcv)

    def test_tp_superieur_sl_plus_de_longs(self) -> None:
        """Avec TP > SL et marché haussier, on attend plus de labels +1 que -1."""
        rng = np.random.default_rng(99)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        # Prix en tendance haussière
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.005, n)))
        high = close * 1.005
        low = close * 0.999
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": np.ones(n)},
            index=dates,
        )
        cfg = LabelingConfig(tp_pct=0.01, sl_pct=0.005, horizon_bars=20)
        labels = apply_triple_barrier(df, cfg)
        valid = labels.dropna()
        n_long = (valid == 1.0).sum()
        n_short = (valid == -1.0).sum()
        # En tendance haussière, TP (plus large) atteint avant SL → plus de longs
        assert n_long >= n_short, f"Marché haussier : n_long={n_long}, n_short={n_short}"
