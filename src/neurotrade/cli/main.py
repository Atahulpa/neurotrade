"""CLI NeuroTrade — points d'entrée Typer.

Aucune logique ici : parse les arguments et délègue à experiments/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from neurotrade.config import load_config
from neurotrade.experiments import ExperimentRunner

app = typer.Typer(
    name="neurotrade",
    help="Système de recherche quantitative reproductible.",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@app.command()
def backtest(
    config_path: Path = typer.Option(
        Path("configs/base.yaml"),
        "--config",
        "-c",
        help="Fichier YAML de configuration.",
        exists=True,
        readable=True,
    ),
    base_path: Path | None = typer.Option(
        None,
        "--base",
        "-b",
        help="YAML de base à fusionner avec --config.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Lance un backtest walk-forward complet."""
    _setup_logging(verbose)
    cfg = load_config(config_path, base_path)
    runner = ExperimentRunner(cfg)
    metrics = runner.run()
    typer.echo(f"Run '{cfg.meta.run_name}' terminé — Sharpe net : {metrics.sharpe_ratio:.3f}")


if __name__ == "__main__":
    app()
