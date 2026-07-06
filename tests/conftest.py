"""Fixtures pytest partagées entre tous les tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurotrade.config.schema import AppConfig, DataConfig


@pytest.fixture
def default_config() -> AppConfig:
    """Configuration par défaut (valeurs Pydantic, sans lire de fichier)."""
    return AppConfig()


@pytest.fixture
def base_yaml_path() -> Path:
    """Chemin du YAML de référence."""
    return Path(__file__).parent.parent / "configs" / "base.yaml"


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Série OHLCV synthétique reproductible (1 000 bougies 1-min).

    Marche aléatoire log-normale autour de 50 000 avec volume gaussien.
    Utilisé dans tous les tests qui ont besoin de données de marché
    sans déclencher de téléchargement réseau.
    """
    rng = np.random.default_rng(42)
    n = 1_000
    dates = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

    log_ret = rng.normal(0.0, 0.001, n)
    close = 50_000.0 * np.exp(np.cumsum(log_ret))
    noise_h = np.abs(rng.normal(0.0, 0.0005, n))
    noise_l = np.abs(rng.normal(0.0, 0.0005, n))
    noise_o = rng.normal(0.0, 0.0003, n)

    high = close * (1 + noise_h)
    low = close * (1 - noise_l)
    open_ = np.roll(close, 1) * (1 + noise_o)
    open_[0] = close[0]
    # Corrige les cas où high < close ou low > close
    high = np.maximum(high, close)
    low = np.minimum(low, close)

    volume = np.abs(rng.normal(100.0, 20.0, n))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def data_config(tmp_path: Path) -> DataConfig:
    """DataConfig pointant vers tmp_path pour les tests de cache."""
    return DataConfig(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        start="2024-01-01",
        end="2024-01-01",
        cache_dir=tmp_path,
    )
