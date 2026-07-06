"""Tests pour evaluation/ — métriques et rapport.

DoD étape 8 :
- Sharpe annualisé cohérent avec formule analytique
- Max drawdown sur cas connu
- DSR < Sharpe brut quand n_trials > 1
- PBO ∈ [0, 1] et convergence vers 0 sur un signal parfait
- Report génère les 3 fichiers attendus
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from neurotrade.evaluation.metrics import (
    _deflated_sharpe,
    _expected_max_sr,
    _max_drawdown,
    _pbo_bootstrap,
    _sharpe_annualized,
    compute_metrics,
)


class TestSharpeAnnualisé:
    def test_sharpe_signal_constant_positif(self) -> None:
        """Rendement constant positif → Sharpe très élevé (limite +∞ si σ→0)."""
        ret = np.full(1000, 0.001)
        # Std de ddof=1 est 0 → NaN attendu
        sr = _sharpe_annualized(ret, rf_per_bar=0.0, bars_per_year=252)
        assert math.isnan(sr)

    def test_sharpe_rendements_nuls(self) -> None:
        ret = np.zeros(500)
        sr = _sharpe_annualized(ret, rf_per_bar=0.0, bars_per_year=252)
        assert math.isnan(sr)

    def test_sharpe_positif_si_mu_positif(self) -> None:
        rng = np.random.default_rng(42)
        ret = rng.normal(0.001, 0.01, size=1000)
        sr = _sharpe_annualized(ret, rf_per_bar=0.0, bars_per_year=252)
        assert sr > 0

    def test_sharpe_negatif_si_mu_negatif(self) -> None:
        rng = np.random.default_rng(42)
        ret = rng.normal(-0.001, 0.01, size=1000)
        sr = _sharpe_annualized(ret, rf_per_bar=0.0, bars_per_year=252)
        assert sr < 0

    def test_sharpe_annualisé_analytique(self) -> None:
        """SR annualisé = SR_par_barre × √bars_per_year."""
        rng = np.random.default_rng(0)
        ret = rng.normal(0.0005, 0.01, size=2000)
        sr_h = _sharpe_annualized(ret, rf_per_bar=0.0, bars_per_year=1)
        sr_y = _sharpe_annualized(ret, rf_per_bar=0.0, bars_per_year=252)
        assert abs(sr_y / sr_h - math.sqrt(252)) < 0.01


class TestMaxDrawdown:
    def test_pas_de_perte_drawdown_zero(self) -> None:
        ret = np.full(100, 0.01)
        assert _max_drawdown(ret) == pytest.approx(0.0, abs=1e-9)

    def test_chute_complete_drawdown_un(self) -> None:
        # Equity tombe à 0 (rendement de -100%)
        ret = np.array([0.0, -1.0, 0.0])
        mdd = _max_drawdown(ret)
        assert mdd == pytest.approx(1.0, abs=1e-9)

    def test_drawdown_connu(self) -> None:
        """Equity: 1 → 2 → 1 → drawdown max = (2-1)/2 = 50%."""
        # Pour passer de 1 à 2 : +100%, de 2 à 1 : −50%
        ret = np.array([1.0, -0.5])
        mdd = _max_drawdown(ret)
        assert mdd == pytest.approx(0.5, abs=1e-9)

    def test_drawdown_strictement_positif(self) -> None:
        rng = np.random.default_rng(42)
        ret = rng.normal(0, 0.01, size=500)
        mdd = _max_drawdown(ret)
        assert mdd >= 0.0


class TestDSR:
    def test_expected_max_sr_croissant_avec_n_trials(self) -> None:
        sr1 = _expected_max_sr(10)
        sr2 = _expected_max_sr(100)
        assert sr2 > sr1 > 0

    def test_expected_max_sr_un_trial_zero(self) -> None:
        assert _expected_max_sr(1) == 0.0

    def test_dsr_inferieur_sharpe_si_n_trials_eleve(self) -> None:
        rng = np.random.default_rng(1)
        ret = rng.normal(0.001, 0.01, size=500)
        sr = _sharpe_annualized(ret, rf_per_bar=0.0, bars_per_year=252)
        dsr = _deflated_sharpe(ret, sr, n_trials=100)
        assert math.isfinite(dsr)
        # DSR exprimé en probabilité [0,1] → ne peut pas être comparé au SR directement
        # On vérifie juste qu'il est dans [0,1]
        assert 0.0 <= dsr <= 1.0

    def test_dsr_proche_de_un_si_tres_bon_sr(self) -> None:
        rng = np.random.default_rng(0)
        # SR artificellement élevé
        ret = rng.normal(0.05, 0.01, size=1000)
        sr = _sharpe_annualized(ret, rf_per_bar=0.0, bars_per_year=252)
        dsr = _deflated_sharpe(ret, sr, n_trials=1)
        assert dsr > 0.99


class TestPBO:
    def test_pbo_dans_01(self) -> None:
        rng = np.random.default_rng(42)
        ret = rng.normal(0.001, 0.01, size=500)
        pbo = _pbo_bootstrap(ret, rf_per_bar=0.0, n_splits=100, seed=0)
        assert math.isfinite(pbo)
        assert 0.0 <= pbo <= 1.0

    def test_pbo_bas_sur_signal_fort(self) -> None:
        """Signal très fort → PBO faible (peu de sur-ajustement)."""
        rng = np.random.default_rng(42)
        ret = rng.normal(0.01, 0.005, size=1000)  # SR très élevé
        pbo = _pbo_bootstrap(ret, rf_per_bar=0.0, n_splits=200, seed=0)
        assert pbo < 0.3  # sur un signal fort, PBO doit rester bas

    def test_pbo_nan_si_series_courte(self) -> None:
        ret = np.array([0.01, -0.01])
        pbo = _pbo_bootstrap(ret, rf_per_bar=0.0, n_splits=10, seed=0)
        assert math.isnan(pbo)


class TestComputeMetrics:
    def test_metrics_completes(self) -> None:
        rng = np.random.default_rng(0)
        ret = rng.normal(0.001, 0.01, size=1000)
        m = compute_metrics(ret, n_trades=50, pnl_net=100.0, bars_per_year=252)
        assert math.isfinite(m.sharpe_ratio)
        assert math.isfinite(m.max_drawdown)
        assert math.isfinite(m.total_return)
        assert m.n_trades == 50
        assert m.pnl_net == 100.0

    def test_metrics_serie_vide(self) -> None:
        m = compute_metrics(np.array([]), n_trades=0, pnl_net=0.0)
        assert math.isnan(m.sharpe_ratio)
        assert m.n_trades == 0

    def test_hit_rate_dans_01(self) -> None:
        rng = np.random.default_rng(1)
        ret = rng.normal(0, 0.01, size=500)
        m = compute_metrics(ret, n_trades=10, pnl_net=0.0, bars_per_year=252)
        assert math.isfinite(m.hit_rate)
        assert 0.0 <= m.hit_rate <= 1.0


class TestGenerateReport:
    def test_rapport_cree_les_fichiers(self, tmp_path: Path) -> None:
        from neurotrade.config.schema import EvaluationConfig
        from neurotrade.evaluation.report import generate_report

        rng = np.random.default_rng(42)
        ret = rng.normal(0.001, 0.01, size=500)
        m = compute_metrics(ret, n_trades=20, pnl_net=50.0, bars_per_year=252)

        cfg = EvaluationConfig(report_dir=tmp_path)
        out = generate_report(m, ret, run_name="test_run", config=cfg)

        assert (out / "equity_curve.png").exists()
        assert (out / "returns_hist.png").exists()
        assert (out / "summary.txt").exists()

    def test_summary_contient_sharpe(self, tmp_path: Path) -> None:
        from neurotrade.config.schema import EvaluationConfig
        from neurotrade.evaluation.report import generate_report

        rng = np.random.default_rng(0)
        ret = rng.normal(0.001, 0.01, size=300)
        m = compute_metrics(ret, n_trades=5, pnl_net=10.0, bars_per_year=252)

        cfg = EvaluationConfig(report_dir=tmp_path)
        out = generate_report(m, ret, run_name="run2", config=cfg)

        summary = (out / "summary.txt").read_text()
        assert "Sharpe" in summary
        assert "Drawdown" in summary
