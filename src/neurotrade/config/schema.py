"""Schémas Pydantic pour la configuration NeuroTrade.

Chaque section du YAML correspond à un modèle. AppConfig agrège le tout.
Aucune logique métier ici — uniquement validation et valeurs par défaut.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MetaConfig(BaseModel):
    """Métadonnées du run (seed, identifiant, version)."""

    version: str = "0.1.0"
    seed: int = Field(42, ge=0, description="Seed global pour la reproductibilité.")
    run_name: str = "baseline"


class DataConfig(BaseModel):
    """Source de données OHLCV et cache parquet."""

    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    start: str = "2023-01-01"
    end: str = "2024-12-31"
    cache_dir: Path = Path("data/")
    max_gap_bars: int = Field(5, ge=0)
    timezone: str = "UTC"

    @field_validator("timeframe")
    @classmethod
    def timeframe_valide(cls, v: str) -> str:
        valides = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if v not in valides:
            raise ValueError(f"timeframe doit être parmi {valides}, reçu : {v!r}")
        return v


class FeaturesConfig(BaseModel):
    """Paramètres du pipeline de features."""

    # Log-returns calculés sur plusieurs horizons (en nombre de bougies)
    horizons: list[int] = Field(default=[1, 5, 15, 60], min_length=1)
    vol_window: int = Field(20, ge=2, description="Fenêtre volatilité réalisée.")
    momentum_window: int = Field(30, ge=2, description="Fenêtre z-score momentum.")
    use_volume: bool = True
    # Garde-fou : si le pipeline produit plus de max_features colonnes → erreur
    max_features: int = Field(10, ge=1)

    @field_validator("horizons")
    @classmethod
    def horizons_positifs(cls, v: list[int]) -> list[int]:
        if any(h <= 0 for h in v):
            raise ValueError("Tous les horizons doivent être > 0.")
        return sorted(v)


class LabelingConfig(BaseModel):
    """Paramètres du labeling triple-barrier."""

    tp_pct: float = Field(0.002, gt=0, description="Take-profit en fraction de prix.")
    sl_pct: float = Field(0.001, gt=0, description="Stop-loss en fraction de prix.")
    # Horizon max avant d'attribuer le label neutre (0)
    horizon_bars: int = Field(60, ge=1)
    # Seuil minimum d'edge prédit pour autoriser un trade (net de coûts)
    min_ret_to_trade: float = Field(0.0005, ge=0)


class ModelConfig(BaseModel):
    """Architecture et hyperparamètres d'entraînement."""

    arch: Literal["mlp", "cnn1d", "lstm"] = "mlp"
    hidden_dims: list[int] = Field(
        default=[64, 32],
        description="Dimensions des couches cachées MLP (ignoré pour cnn1d/lstm).",
    )
    dropout: float = Field(0.3, ge=0.0, le=1.0)
    weight_decay: float = Field(1e-4, ge=0.0)
    lr: float = Field(1e-3, gt=0.0)
    batch_size: int = Field(256, ge=1)
    max_epochs: int = Field(100, ge=1)
    patience: int = Field(10, ge=1, description="Early stopping : patience en epochs.")
    # Nombre de bougies consécutives par échantillon (1=MLP flat, >1=CNN/LSTM séquence)
    window_size: int = Field(1, ge=1)


class BacktestConfig(BaseModel):
    """Paramètres du walk-forward et de la séparation train/val/test."""

    # Longueur de la fenêtre d'entraînement en bougies
    train_window_bars: int = Field(21600, ge=100)
    # Pas de glissement entre deux fenêtres consécutives
    step_bars: int = Field(1440, ge=1)
    # Fraction de la fenêtre train réservée à la validation in-sample
    val_ratio: float = Field(0.2, gt=0.0, lt=1.0)
    # Purge : doit être >= horizon_bars pour éviter toute fuite de label
    purge_bars: int = Field(60, ge=0)
    # Embargo : bougies à ignorer juste après val (délai d'exécution simulé)
    embargo_bars: int = Field(10, ge=0)


class ExecutionConfig(BaseModel):
    """Simulateur d'exécution : coûts complets."""

    fee_maker: float = Field(0.0002, ge=0.0, description="Fraction de frais maker.")
    fee_taker: float = Field(0.0004, ge=0.0, description="Fraction de frais taker.")
    slippage_bps: float = Field(1.0, ge=0.0, description="Slippage en points de base.")
    # Funding rate toutes les 8 h pour les contrats perpétuels
    funding_rate_8h: float = Field(0.0001, ge=0.0)
    # L'ordre est exécuté sur la bougie suivante (pas sur la bougie courante)
    latency_bars: int = Field(1, ge=1)


class RiskConfig(BaseModel):
    """Gestion du risque : sizing, stop journalier, kill-switch."""

    max_position_pct: float = Field(0.02, gt=0.0, le=1.0)
    # Coupe-circuit journalier : si le drawdown dépasse ce seuil, flat + pause
    daily_stop_pct: float = Field(0.01, gt=0.0)
    # Kill-switch volatilité : z-score > seuil → flat + pause
    vol_zscore_threshold: float = Field(3.0, gt=0.0)
    # Kill-switch spread : z-score > seuil → flat + pause
    spread_zscore_threshold: float = Field(4.0, gt=0.0)


class EvaluationConfig(BaseModel):
    """Rapports et métriques."""

    report_dir: Path = Path("reports/")
    # En dessous de ce nombre de trades, les métriques ne sont pas fiables
    min_trades: int = Field(30, ge=1)
    risk_free_rate: float = Field(0.04, ge=0.0, description="Taux sans risque annualisé.")


class AppConfig(BaseModel):
    """Configuration complète de l'application. Chargée depuis un YAML."""

    meta: MetaConfig = Field(default_factory=MetaConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
