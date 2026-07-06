"""Tests pour features/ — pipeline + scaler.

Le test anti-fuite scaler est le DoD critique de l'étape 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurotrade.config.schema import FeaturesConfig
from neurotrade.features.momentum import momentum_zscore
from neurotrade.features.pipeline import build_features, feature_names
from neurotrade.features.returns import log_returns
from neurotrade.features.scaler import FeatureScaler
from neurotrade.features.volatility import realized_volatility
from neurotrade.features.volume import volume_features

# ── Tests des composants individuels ─────────────────────────────────────────


class TestLogReturns:
    def test_shape(self, synthetic_ohlcv: pd.DataFrame) -> None:
        horizons = [1, 5, 15]
        result = log_returns(synthetic_ohlcv["close"], horizons)
        assert result.shape == (len(synthetic_ohlcv), 3)

    def test_noms_colonnes(self, synthetic_ohlcv: pd.DataFrame) -> None:
        result = log_returns(synthetic_ohlcv["close"], [1, 5])
        assert list(result.columns) == ["ret_1", "ret_5"]

    def test_premiere_ligne_nan(self, synthetic_ohlcv: pd.DataFrame) -> None:
        result = log_returns(synthetic_ohlcv["close"], [5])
        assert result["ret_5"].iloc[:5].isna().all()

    def test_valeurs_cohérentes(self) -> None:
        """log_return(t, 1) = log(close[t] / close[t-1])."""
        prices = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = log_returns(prices, [1])
        expected = np.log(110.0 / 100.0)
        assert result["ret_1"].iloc[1] == pytest.approx(expected, rel=1e-6)


class TestVolatility:
    def test_positif(self, synthetic_ohlcv: pd.DataFrame) -> None:
        vol = realized_volatility(synthetic_ohlcv["close"], window=20)
        assert (vol.dropna() >= 0).all()

    def test_nan_au_debut(self, synthetic_ohlcv: pd.DataFrame) -> None:
        vol = realized_volatility(synthetic_ohlcv["close"], window=20)
        assert vol.iloc[:20].isna().all()


class TestMomentum:
    def test_zscore_approximativement_centré(self, synthetic_ohlcv: pd.DataFrame) -> None:
        z = momentum_zscore(synthetic_ohlcv["close"], window=50)
        valid = z.dropna()
        assert abs(valid.mean()) < 1.0  # pas strictement 0 car prix non stationnaire

    def test_nan_au_debut(self, synthetic_ohlcv: pd.DataFrame) -> None:
        z = momentum_zscore(synthetic_ohlcv["close"], window=30)
        # rolling(30) nécessite 30 points → premières 29 valeurs sont NaN
        assert z.iloc[:29].isna().all()


class TestVolumeFeatures:
    def test_colonnes(self, synthetic_ohlcv: pd.DataFrame) -> None:
        vf = volume_features(synthetic_ohlcv["volume"])
        assert list(vf.columns) == ["vol_rel", "log_vol"]

    def test_log_vol_positif(self, synthetic_ohlcv: pd.DataFrame) -> None:
        vf = volume_features(synthetic_ohlcv["volume"])
        assert (vf["log_vol"].dropna() >= 0).all()


# ── Tests du pipeline ─────────────────────────────────────────────────────────


class TestPipeline:
    def test_shape_sans_volume(self, synthetic_ohlcv: pd.DataFrame) -> None:
        cfg = FeaturesConfig(
            horizons=[1, 5], vol_window=20, momentum_window=30, use_volume=False
        )
        X = build_features(synthetic_ohlcv, cfg)
        # 2 horizons + vol + momentum = 4 features, pas de NaN
        assert X.ndim == 2
        assert X.shape[1] == 4
        assert not np.isnan(X).any()

    def test_shape_avec_volume(self, synthetic_ohlcv: pd.DataFrame) -> None:
        cfg = FeaturesConfig(
            horizons=[1, 5], vol_window=20, momentum_window=30, use_volume=True
        )
        X = build_features(synthetic_ohlcv, cfg)
        # 2 horizons + vol + momentum + vol_rel + log_vol = 6 features
        assert X.shape[1] == 6

    def test_feature_names_coheérents(self) -> None:
        cfg = FeaturesConfig(horizons=[1, 5], use_volume=True)
        names = feature_names(cfg)
        assert names[0] == "ret_1"
        assert "vol" in names
        assert "momentum" in names
        assert "vol_rel" in names

    def test_max_features_declenche_erreur(self, synthetic_ohlcv: pd.DataFrame) -> None:
        cfg = FeaturesConfig(
            horizons=[1, 5, 15, 60], use_volume=True, max_features=5
        )
        with pytest.raises(ValueError, match="max_features"):
            build_features(synthetic_ohlcv, cfg)

    def test_dtype_float64(self, synthetic_ohlcv: pd.DataFrame) -> None:
        cfg = FeaturesConfig(horizons=[1], vol_window=20, momentum_window=30, use_volume=False)
        X = build_features(synthetic_ohlcv, cfg)
        assert X.dtype == np.float64


# ── TEST ANTI-FUITE SCALER (DoD étape 2) ──────────────────────────────────────


class TestFeatureScalerAntiFuite:
    """Vérifie que la normalisation ne fuit pas val→train.

    Principe : on calibre le scaler sur une distribution (μ=0, σ=1),
    puis on transforme une distribution différente (μ=5, σ=2).
    Après transformation par les stats de train, la val doit avoir μ ≠ 0.
    Si μ_val_scaled ≈ 0, cela indiquerait que le scaler a "vu" la val pendant fit().
    """

    def test_fit_uniquement_sur_train(self) -> None:
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(500, 5))
        X_val = rng.normal(loc=5.0, scale=2.0, size=(100, 5))  # distribution différente

        scaler = FeatureScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Sur train : μ ≈ 0, σ ≈ 1 (par construction du StandardScaler)
        assert abs(X_train_scaled.mean()) < 0.1, "Train mean devrait être ≈ 0"
        assert abs(X_train_scaled.std() - 1.0) < 0.1, "Train std devrait être ≈ 1"

        # Sur val : μ doit être ≠ 0 car les stats sont celles de train, pas de val
        # Si le scaler avait fuité sur val, X_val_scaled.mean() serait ≈ 0 aussi.
        val_mean = abs(X_val_scaled.mean())
        assert val_mean > 1.0, (
            f"Fuite détectée : val_mean_scaled = {val_mean:.4f} ≈ 0 → "
            "le scaler a probablement utilisé les stats de val pendant fit()."
        )

    def test_transform_avant_fit_leve_erreur(self) -> None:
        scaler = FeatureScaler()
        X = np.ones((10, 3))
        with pytest.raises(RuntimeError, match=r"fit\(\)"):
            scaler.transform(X)

    def test_fit_transform_equivalent(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        scaler = FeatureScaler()
        result_chain = scaler.fit(X).transform(X)
        scaler2 = FeatureScaler()
        result_shortcut = scaler2.fit_transform(X)
        np.testing.assert_array_almost_equal(result_chain, result_shortcut)

    def test_feature_constante_ne_divise_pas_par_zero(self) -> None:
        X_train = np.ones((50, 3))
        scaler = FeatureScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_train)
        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()
