"""Tests pour config/ — schémas Pydantic et chargement YAML.

Ces tests constituent le DoD de l'étape 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurotrade.config import load_config
from neurotrade.config.schema import AppConfig, DataConfig, FeaturesConfig


class TestAppConfigDefaults:
    """Les valeurs par défaut doivent être cohérentes entre elles."""

    def test_instanciation_sans_arguments(self) -> None:
        cfg = AppConfig()
        assert cfg.meta.seed == 42

    def test_seed_positif(self) -> None:
        cfg = AppConfig()
        assert cfg.meta.seed >= 0

    def test_purge_coherent_avec_horizon(self) -> None:
        """purge_bars doit être >= horizon_bars pour éviter la fuite de label."""
        cfg = AppConfig()
        assert cfg.backtest.purge_bars >= cfg.labeling.horizon_bars, (
            f"purge_bars={cfg.backtest.purge_bars} < horizon_bars={cfg.labeling.horizon_bars} "
            "→ risque de fuite de label entre train et val"
        )

    def test_tp_superieur_sl(self) -> None:
        """TP > SL donne un ratio risk/reward > 1 (règle de bonne pratique)."""
        cfg = AppConfig()
        assert cfg.labeling.tp_pct > cfg.labeling.sl_pct

    def test_horizons_tries(self) -> None:
        cfg = AppConfig()
        assert cfg.features.horizons == sorted(cfg.features.horizons)


class TestDataConfigValidation:
    def test_timeframe_invalide_leve_erreur(self) -> None:
        with pytest.raises(ValueError, match="timeframe"):
            DataConfig(timeframe="3m")  # 3m n'est pas dans la liste

    def test_timeframe_valide_accepte(self) -> None:
        cfg = DataConfig(timeframe="5m")
        assert cfg.timeframe == "5m"


class TestFeaturesConfigValidation:
    def test_horizons_negatifs_interdits(self) -> None:
        with pytest.raises(ValueError):
            FeaturesConfig(horizons=[-1, 5])

    def test_horizons_tries_apres_validation(self) -> None:
        cfg = FeaturesConfig(horizons=[60, 1, 15, 5])
        assert cfg.horizons == [1, 5, 15, 60]


class TestLoadConfig:
    def test_charge_base_yaml(self, base_yaml_path: Path) -> None:
        cfg = load_config(base_yaml_path)
        assert isinstance(cfg, AppConfig)
        assert cfg.meta.run_name == "baseline"

    def test_charge_override_avec_base(self, base_yaml_path: Path, tmp_path: Path) -> None:
        override = tmp_path / "override.yaml"
        override.write_text("meta:\n  run_name: test_override\n")
        cfg = load_config(override, base_path=base_yaml_path)
        assert cfg.meta.run_name == "test_override"
        # Les valeurs non overridées viennent de base.yaml
        assert cfg.meta.seed == 42

    def test_fichier_vide_donne_config_par_defaut(self, tmp_path: Path) -> None:
        vide = tmp_path / "empty.yaml"
        vide.write_text("")
        cfg = load_config(vide)
        assert isinstance(cfg, AppConfig)
