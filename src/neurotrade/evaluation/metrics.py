"""Métriques de performance du backtest.

Toutes les fonctions sont pures (numpy + scipy, zéro I/O).
Formules de référence :
  - Sharpe annualisé : SR = (μ - rf/bars_per_year) / σ × √bars_per_year
  - DSR (Deflated Sharpe Ratio) : Bailey, Borwein, Lopez de Prado & Zhu (2014)
      corrige le biais de sélection lorsque n_trials > 1
  - PBO (Probability of Backtest Overfitting) : Bailey et al. (2014)
      estimé par bootstrap aléatoire IS/OOS 50-50
  - Max Drawdown : perte maximale de pic à creux sur la courbe equity
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Constante d'Euler-Mascheroni (pour DSR)
_EULER_MASCHERONI = 0.5772156649


@dataclass
class BacktestMetrics:
    """Résultats complets d'un backtest pour un run."""

    sharpe_ratio: float
    deflated_sharpe_ratio: float   # DSR : corrige le biais de sélection multiple
    pbo: float                     # Probability of Backtest Overfitting ∈ [0, 1]
    max_drawdown: float            # positif, ex. 0.15 = −15%
    total_return: float            # rendement total cumulé
    annualized_return: float
    annualized_volatility: float
    hit_rate: float                # fraction des barres actives avec rendement > 0
    n_trades: int
    turnover: float                # trades par an
    pnl_net: float                 # PnL total net de coûts


def compute_metrics(
    returns: np.ndarray,
    n_trades: int,
    pnl_net: float,
    risk_free_rate: float = 0.04,
    n_trials: int = 1,
    bars_per_year: int = 8760,
    pbo_n_splits: int = 200,
    pbo_seed: int = 0,
) -> BacktestMetrics:
    """Calcule toutes les métriques à partir de la série de rendements nets.

    Args:
        returns: Rendements nets par bougie (float64, peut contenir des zéros).
        n_trades: Nombre total de trades exécutés.
        pnl_net: PnL total net en unités de capital.
        risk_free_rate: Taux sans risque annualisé.
        n_trials: Nombre de configurations testées (corrige le biais DSR).
        bars_per_year: Barres par an selon le timeframe (8760=1h, 252=1d, etc.).
        pbo_n_splits: Nombre de splits bootstrap pour estimer le PBO.
        pbo_seed: Seed pour la reproductibilité du PBO.

    Returns:
        BacktestMetrics rempli. Les métriques sont NaN si données insuffisantes.
    """
    if len(returns) < 2:
        return _empty_metrics(n_trades, pnl_net)

    ret = np.asarray(returns, dtype=np.float64)
    mu = float(ret.mean())
    sigma = float(ret.std(ddof=1))
    rf_per_bar = risk_free_rate / bars_per_year

    sr = _sharpe_annualized(ret, rf_per_bar, bars_per_year)
    dsr = _deflated_sharpe(ret, sr, n_trials)
    pbo = _pbo_bootstrap(ret, rf_per_bar, pbo_n_splits, pbo_seed)
    mdd = _max_drawdown(ret)

    ann_return = mu * bars_per_year
    ann_vol = sigma * math.sqrt(bars_per_year)

    equity = np.cumprod(1.0 + ret)
    total_ret = float(equity[-1]) - 1.0

    active = ret != 0.0
    hit_rate = float((ret[active] > 0).mean()) if active.sum() > 0 else float("nan")

    n_years = len(ret) / bars_per_year
    turnover = n_trades / n_years if n_years > 0 else float("nan")

    return BacktestMetrics(
        sharpe_ratio=sr,
        deflated_sharpe_ratio=dsr,
        pbo=pbo,
        max_drawdown=mdd,
        total_return=total_ret,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        hit_rate=hit_rate,
        n_trades=n_trades,
        turnover=turnover,
        pnl_net=pnl_net,
    )


# ── Fonctions internes (pures, testables individuellement) ───────────────────

def _sharpe_annualized(
    returns: np.ndarray,
    rf_per_bar: float,
    bars_per_year: int,
) -> float:
    """Sharpe ratio annualisé, NaN si σ ≈ 0."""
    sigma = float(returns.std(ddof=1))
    if sigma < 1e-12:
        return float("nan")
    mu = float(returns.mean()) - rf_per_bar
    return mu / sigma * math.sqrt(bars_per_year)


def _max_drawdown(returns: np.ndarray) -> float:
    """Drawdown maximum en valeur positive (ex. 0.15 = 15% de perte pic-à-creux)."""
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    # Évite division par zéro si running_max = 0 (improbable mais défensif)
    safe_max = np.where(running_max > 0, running_max, 1.0)
    drawdowns = (running_max - equity) / safe_max
    return float(drawdowns.max())


def _expected_max_sr(n_trials: int) -> float:
    """Sharpe ratio maximal espéré sur n_trials essais indépendants (approximation DSR).

    Formule : (1-γ)×Φ^{-1}(1-1/N) + γ×Φ^{-1}(1-1/(N×e))
    Référence : Bailey et al. (2014), équation (3).
    """
    if n_trials <= 1:
        return 0.0
    inv_cdf = stats.norm.ppf
    gamma = _EULER_MASCHERONI
    term1 = (1 - gamma) * float(inv_cdf(1.0 - 1.0 / n_trials))
    term2 = gamma * float(inv_cdf(1.0 - 1.0 / (n_trials * math.e)))
    return term1 + term2


def _deflated_sharpe(
    returns: np.ndarray,
    sr_hat: float,
    n_trials: int,
) -> float:
    """DSR corrigeant le biais de sélection multiple.

    DSR = Φ[(SR̂ − SR*) / √Var(SR̂)]
    Var(SR̂) = (1 − ρ̂×SR̂ + ((κ̂+2)/4)×SR̂²) / (T−1)
    Référence : Bailey et al. (2014).
    """
    if not math.isfinite(sr_hat):
        return float("nan")
    n = len(returns)
    if n < 4:
        return float("nan")

    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=True))  # excès de kurtosis
    sr_star = _expected_max_sr(n_trials)

    var_sr = (1.0 - skew * sr_hat + ((kurt + 2) / 4) * sr_hat**2) / (n - 1)
    if var_sr <= 0:
        return float("nan")

    z = (sr_hat - sr_star) / math.sqrt(var_sr)
    dsr: float = float(stats.norm.cdf(z))
    return dsr


def _pbo_bootstrap(
    returns: np.ndarray,
    rf_per_bar: float,
    n_splits: int,
    seed: int,
) -> float:
    """Estime le PBO via bootstrap IS/OOS 50-50 (adapté pour une stratégie unique).

    PBO ≈ P(SR_OOS ≤ 0 | SR_IS > 0) mesuré sur n_splits splits aléatoires.
    Cible : PBO << 0.5 indique un backtest robuste.
    """
    n = len(returns)
    if n < 20:
        return float("nan")

    rng = np.random.default_rng(seed)
    overfit = 0
    pos_is = 0
    half = n // 2

    for _ in range(n_splits):
        idx = rng.permutation(n)
        is_r = returns[idx[:half]]
        oos_r = returns[idx[half:]]

        is_sr = _sharpe_annualized(is_r, rf_per_bar, bars_per_year=1)
        oos_sr = _sharpe_annualized(oos_r, rf_per_bar, bars_per_year=1)

        if math.isfinite(is_sr) and is_sr > 0:
            pos_is += 1
            if math.isfinite(oos_sr) and oos_sr <= 0:
                overfit += 1

    if pos_is == 0:
        return float("nan")
    return overfit / pos_is


def _empty_metrics(n_trades: int, pnl_net: float) -> BacktestMetrics:
    nan = float("nan")
    return BacktestMetrics(
        sharpe_ratio=nan,
        deflated_sharpe_ratio=nan,
        pbo=nan,
        max_drawdown=nan,
        total_return=nan,
        annualized_return=nan,
        annualized_volatility=nan,
        hit_rate=nan,
        n_trades=n_trades,
        turnover=nan,
        pnl_net=pnl_net,
    )
