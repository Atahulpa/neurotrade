"""Génération de rapports matplotlib pour chaque run.

Crée un répertoire `report_dir/run_name/` avec :
  - equity_curve.png  : courbe d'equity cumulée + drawdown
  - returns_hist.png  : distribution des rendements par barre
  - summary.txt       : tableau texte des métriques clés
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # mode non-interactif (pas de display requis)
import matplotlib.pyplot as plt

from neurotrade.config.schema import EvaluationConfig

from .metrics import BacktestMetrics

logger = logging.getLogger(__name__)


def generate_report(
    metrics: BacktestMetrics,
    returns: np.ndarray,
    run_name: str,
    config: EvaluationConfig,
) -> Path:
    """Génère les graphiques et sauvegarde le rapport dans report_dir/run_name/.

    Args:
        metrics: Métriques calculées par compute_metrics().
        returns: Série de rendements nets (même longueur que le backtest).
        run_name: Identifiant du run (utilisé comme nom de dossier).
        config: Configuration d'évaluation (report_dir, min_trades, …).

    Returns:
        Chemin du répertoire du rapport créé.
    """
    out_dir = config.report_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ret = np.asarray(returns, dtype=np.float64)

    _plot_equity_curve(ret, metrics, out_dir)
    _plot_returns_hist(ret, out_dir)
    _write_summary(metrics, out_dir)

    logger.info("Rapport généré dans : %s", out_dir)
    return out_dir


# ── Graphiques ───────────────────────────────────────────────────────────────

def _plot_equity_curve(
    returns: np.ndarray,
    metrics: BacktestMetrics,
    out_dir: Path,
) -> None:
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / np.where(running_max > 0, running_max, 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(equity, linewidth=1.0, color="#2563EB", label="Equity curve")
    axes[0].set_ylabel("Valeur du portefeuille")
    axes[0].set_title(
        f"Equity curve — Sharpe={metrics.sharpe_ratio:.2f}  "
        f"DSR={metrics.deflated_sharpe_ratio:.2f}  "
        f"MDD={metrics.max_drawdown:.1%}"
    )
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(range(len(drawdown)), -drawdown, 0,
                         color="#EF4444", alpha=0.6, label="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Barres")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=120)
    plt.close(fig)


def _plot_returns_hist(returns: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    active = returns[returns != 0.0]
    if len(active) > 0:
        ax.hist(active, bins=60, color="#3B82F6", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Rendement net par barre")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des rendements (barres actives)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "returns_hist.png", dpi=120)
    plt.close(fig)


def _write_summary(metrics: BacktestMetrics, out_dir: Path) -> None:
    def fmt(v: float, pct: bool = False) -> str:
        if not (v == v):  # NaN check
            return "N/A"
        return f"{v:.2%}" if pct else f"{v:.4f}"

    lines = [
        "=" * 50,
        "  NEUROTRADE — Résumé du backtest",
        "=" * 50,
        f"  Sharpe Ratio          : {fmt(metrics.sharpe_ratio)}",
        f"  Deflated Sharpe (DSR) : {fmt(metrics.deflated_sharpe_ratio)}",
        f"  PBO                   : {fmt(metrics.pbo, pct=True)}",
        f"  Max Drawdown          : {fmt(metrics.max_drawdown, pct=True)}",
        f"  Rendement total       : {fmt(metrics.total_return, pct=True)}",
        f"  Rendement annualisé   : {fmt(metrics.annualized_return, pct=True)}",
        f"  Volatilité annualisée : {fmt(metrics.annualized_volatility, pct=True)}",
        f"  Hit rate              : {fmt(metrics.hit_rate, pct=True)}",
        f"  Nombre de trades      : {metrics.n_trades}",
        f"  Turnover (trades/an)  : {fmt(metrics.turnover)}",
        f"  PnL net               : {fmt(metrics.pnl_net)}",
        "=" * 50,
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
