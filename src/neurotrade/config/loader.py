"""Chargement de la configuration depuis un fichier YAML.

Stratégie : charger base.yaml, fusionner avec le fichier override, valider avec Pydantic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .schema import AppConfig

logger = logging.getLogger(__name__)


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Fusionne `override` dans `base` de façon récursive (override gagne)."""
    result: dict[str, object] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def load_config(path: Path, base_path: Path | None = None) -> AppConfig:
    """Charge et valide la configuration.

    Si `base_path` est fourni, fusionne base → override.
    Sinon, charge `path` seul (doit être complet ou s'appuyer sur les valeurs par défaut).

    Args:
        path: Fichier YAML principal (ou override si base_path est fourni).
        base_path: Fichier YAML de base optionnel.

    Returns:
        AppConfig validé par Pydantic.
    """
    with open(path) as f:
        raw: dict[str, object] = yaml.safe_load(f) or {}

    if base_path is not None:
        with open(base_path) as f:
            base_raw: dict[str, object] = yaml.safe_load(f) or {}
        raw = _deep_merge(base_raw, raw)

    config = AppConfig.model_validate(raw)
    logger.info("Configuration chargée depuis %s (run_name=%r)", path, config.meta.run_name)
    return config
