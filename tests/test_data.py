"""Tests pour data/ — cache, resample, gap detection.

Le fetcher (réseau) est testé via mock. Pas de téléchargement réel.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from neurotrade.config.schema import DataConfig
from neurotrade.data.cache import cache_path, load_from_cache, save_to_cache
from neurotrade.data.resample import ensure_utc, resample_ohlcv


class TestCache:
    def test_round_trip(self, synthetic_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
        """Sauvegarde puis rechargement doit reproduire le DataFrame exactement."""
        path = tmp_path / "test.parquet"
        save_to_cache(synthetic_ohlcv, path)
        loaded = load_from_cache(path)

        assert list(loaded.columns) == list(synthetic_ohlcv.columns)
        assert len(loaded) == len(synthetic_ohlcv)
        assert loaded.index.tz is not None
        assert str(loaded.index.tz) == "UTC"
        pd.testing.assert_frame_equal(loaded, synthetic_ohlcv, check_freq=False)

    def test_load_manquant_leve_erreur(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_from_cache(tmp_path / "inexistant.parquet")

    def test_cache_path_inclut_symbol_et_dates(self, data_config: DataConfig) -> None:
        p = cache_path(data_config)
        assert "BTC_USDT" in p.name
        assert "1m" in p.name
        assert "2024-01-01" in p.name

    def test_save_cree_repertoire(self, synthetic_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
        nested = tmp_path / "sous" / "dossier" / "data.parquet"
        save_to_cache(synthetic_ohlcv, nested)
        assert nested.exists()


class TestResample:
    def test_resample_1min_vers_5min(self, synthetic_ohlcv: pd.DataFrame) -> None:
        r = resample_ohlcv(synthetic_ohlcv, "5m")
        # 1 000 bougies de 1 min → ~200 bougies de 5 min
        assert len(r) == pytest.approx(200, abs=2)
        assert (r["high"] >= r["close"]).all()
        assert (r["low"] <= r["close"]).all()
        assert (r["volume"] > 0).all()

    def test_resample_conserve_high_max(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """Le high d'une bougie 5 min doit être le max des 5 bougies 1 min."""
        r = resample_ohlcv(synthetic_ohlcv, "5m")
        first_bucket_high = synthetic_ohlcv["high"].iloc[:5].max()
        assert r["high"].iloc[0] == pytest.approx(first_bucket_high)

    def test_resample_timeframe_invalide(self, synthetic_ohlcv: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            resample_ohlcv(synthetic_ohlcv, "3m")

    def test_resample_1min_vers_1h(self, synthetic_ohlcv: pd.DataFrame) -> None:
        r = resample_ohlcv(synthetic_ohlcv, "1h")
        # 1 000 bougies de 1 min → ~16 ou 17 bougies de 1 h
        assert 15 <= len(r) <= 18


class TestEnsureUTC:
    def test_index_sans_tz_localize(self, synthetic_ohlcv: pd.DataFrame) -> None:
        df_naive = synthetic_ohlcv.copy()
        df_naive.index = df_naive.index.tz_localize(None)
        result = ensure_utc(df_naive)
        assert str(result.index.tz) == "UTC"

    def test_index_autre_tz_converti(self, synthetic_ohlcv: pd.DataFrame) -> None:
        df_paris = synthetic_ohlcv.copy()
        df_paris.index = df_paris.index.tz_convert("Europe/Paris")
        result = ensure_utc(df_paris)
        assert str(result.index.tz) == "UTC"

    def test_index_deja_utc_inchange(self, synthetic_ohlcv: pd.DataFrame) -> None:
        result = ensure_utc(synthetic_ohlcv)
        assert str(result.index.tz) == "UTC"
        assert len(result) == len(synthetic_ohlcv)


class TestFetcherMock:
    """Teste la logique du fetcher sans appels réseau réels."""

    def _make_fake_batch(self, start_ms: int, n: int, tf_ms: int) -> list[list[object]]:
        """Génère une fausse réponse ccxt : liste de [ts, o, h, l, c, v]."""
        return [
            [start_ms + i * tf_ms, 50000.0, 50100.0, 49900.0, 50050.0, 100.0]
            for i in range(n)
        ]

    def test_fetch_pagine_correctement(self, data_config: DataConfig) -> None:
        """Le fetcher doit combiner plusieurs pages en un seul DataFrame."""
        tf_ms = 60_000
        batch1 = self._make_fake_batch(
            int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000), 1000, tf_ms
        )
        batch2 = self._make_fake_batch(batch1[-1][0] + tf_ms, 500, tf_ms)  # type: ignore[operator]

        mock_exchange = MagicMock()
        mock_exchange.parse8601.side_effect = lambda s: int(
            pd.Timestamp(s, tz="UTC").timestamp() * 1000
        )
        mock_exchange.fetch_ohlcv.side_effect = [batch1, batch2, []]

        with patch("neurotrade.data.fetcher.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = mock_exchange
            from neurotrade.data.fetcher import fetch_ohlcv

            cfg = DataConfig(
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                start="2024-01-01",
                end="2024-01-02",
            )
            df = fetch_ohlcv(cfg)

        assert len(df) == 1500
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.tz is not None

    def test_fetch_vide_leve_erreur(self, data_config: DataConfig) -> None:
        """Si l'exchange ne renvoie rien, ValueError doit être levé."""
        mock_exchange = MagicMock()
        mock_exchange.parse8601.side_effect = lambda s: int(
            pd.Timestamp(s, tz="UTC").timestamp() * 1000
        )
        mock_exchange.fetch_ohlcv.return_value = []

        with patch("neurotrade.data.fetcher.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = mock_exchange
            from neurotrade.data.fetcher import fetch_ohlcv

            with pytest.raises(ValueError, match="Aucune donnée"):
                fetch_ohlcv(data_config)
