# CLAUDE.md — Conventions NeuroTrade

## Objectif
Système de recherche quantitative reproductible : réseau de neurones maison (PyTorch),
walk-forward adaptatif, backtest réaliste net de coûts. Priorité : robustesse, pas Sharpe max.

## Stack
- Python 3.11+, `uv` + `pyproject.toml`, layout `src/`
- PyTorch (pas de LLM, pas de NLP, données de marché uniquement)
- Pydantic v2 + YAML pour la config
- Typer CLI, logging structuré (jamais de `print`)
- Ruff + Black + mypy strict + pytest

## Architecture — couches séparées, dépendances à sens unique
```
config/ ← data/ ← features/ ← labeling/ ← models/ ← backtest/ ← evaluation/
                                                                 ↑
                                                            experiments/
```
- `features/`, `labeling/`, `evaluation/` : **cœurs purs** — zéro I/O, zéro réseau, zéro torch.
- `data/` : seule couche qui touche disque/réseau.
- `models/` : seule couche qui dépend de torch.
- Si une couche a besoin d'une autre couche de même niveau → poser la question avant.

## Règles de code
1. Type hints partout, mypy strict.
2. Aucun `print` — toujours `logging`.
3. Seeds fixés (numpy + random + torch) via `meta.seed` dans la config.
4. Normalisation `fit` sur train uniquement (test anti-fuite obligatoire).
5. Purge + embargo entre train/val/test.
6. Commentaires = le *pourquoi*, jamais le *quoi*.
7. Tout en français : commentaires, docstrings, logs, docs. Noms de code en anglais.

## Méthode quant (non négociable)
- Walk-forward glissant : cadence ré-entraînement ≠ fenêtre d'entraînement.
- Labeling triple-barrier (TP/SL/horizon).
- Simulateur d'exécution : frais maker/taker, slippage, funding, latence.
- Trade uniquement si edge prédit > coûts + marge.
- Métriques : Sharpe, Deflated Sharpe (DSR), PBO, drawdown, turnover, hit-rate.
- Kill-switch : flat + pause si vol/spread anormaux.

## Phasage
- **Phase 0** ✅ Échafaudage (en cours)
- **Phase 1** Recherche offline (étapes 1–9)
- **Phase 2** Live/paper — démarrage uniquement sur demande explicite

## Commandes utiles
```bash
make check    # lint + types + tests (tout d'un coup)
make fmt      # formatage automatique
make test     # tests seuls
uv run neurotrade --help
```
