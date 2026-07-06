# NeuroTrade

> **Avertissement** : la grande majorité des stratégies algo retail perdent net de frais.
> Ce système est conçu pour *ne pas se mentir* : backtest réaliste, métriques honnêtes,
> paper trading prolongé avant tout capital réel. **Aucun ordre réel en Phase 1.**

Système de **recherche quantitative reproductible** : réseau de neurones maison (PyTorch),
walk-forward adaptatif, simulateur d'exécution complet, métriques DSR/PBO.

## Architecture

```
neurotrade/
├── config/      ← Schémas Pydantic + chargement YAML
├── data/        ← Fetch OHLCV + cache Parquet (seule couche I/O)
├── features/    ← Feature engineering pur (numpy/pandas, zéro I/O)
├── labeling/    ← Triple-barrier (cœur pur)
├── models/      ← MLP / 1D-CNN / LSTM + entraînement (seule couche torch)
├── backtest/    ← Walk-forward + simulateur exécution + risque
├── evaluation/  ← Métriques + rapports (cœur pur)
├── experiments/ ← Orchestration des runs
└── cli/         ← Points d'entrée Typer
```

Règle fondamentale : `features/`, `labeling/`, `evaluation/` n'importent jamais
`data/`, `models/` ni torch. Dépendances à sens unique uniquement.

## Installation

```bash
# Prérequis : uv installé (https://docs.astral.sh/uv/)
uv sync
```

## Lancer un backtest

```bash
uv run neurotrade backtest --config configs/btc_1min.yaml
```

## Lire un rapport

Les rapports sont générés dans `reports/<run_name>/`. Chaque run produit :
- `metrics.json` — toutes les métriques (Sharpe, DSR, PBO, drawdown…)
- `equity_curve.png` — courbe des gains nets
- `walk_forward.png` — performance fenêtre par fenêtre

## Qualité

```bash
make check   # lint (ruff) + types (mypy strict) + tests (pytest)
make fmt     # formatage automatique (black + ruff --fix)
```

## Roadmap

- **Phase 0** ✅ Échafaudage, outillage qualité
- **Phase 1** 🔄 Recherche offline (étapes 1–9)
  1. `data/` — fetch OHLCV + cache
  2. `features/` — pipeline + anti-fuite scaler
  3. `labeling/` — triple-barrier
  4. `models/` — MLP, 1D-CNN, LSTM
  5. `backtest/` — walk-forward + purge/embargo
  6. `backtest/` — simulateur d'exécution
  7. `backtest/` — risque + kill-switch
  8. `evaluation/` — métriques + rapports
  9. `experiments/` — robustesse cross-asset
- **Phase 2** ⬜ Paper trading (uniquement après validation Phase 1)

Phase 1 complète. Voici le résumé de tout ce qui a été implémenté :

Phase 1 — tous les 9 steps sont terminés et commités :

Étape	Commit	Tests	Highlights
4 — models/	a70bcca	7	MLP/CNN1D/LSTM + Trainer early stopping (delta=1e-4), mypy strict
5 — walk_forward	072c394	15	Purge gap exact + embargo gap exact, no-overlap sur 3000 bars
6 — execution	072c394	7	PnL vérifié analytiquement (0 fees, slippage directionnel, reversal)
7 — risk	072c394	14	Sizing, stop journalier, kill-switch z-score, std=0 safe
8 — evaluation	019b057	21	Sharpe, DSR (Bailey 2014), PBO bootstrap, rapport matplotlib
9 — runner	019b057	—	Pipeline complet data→eval, neurotrade backtest opérationnel
État global : 116 tests (95 rapides + 21 évaluation), mypy --strict + ruff propres sur 35 fichiers. Phase 2 (live/paper) attend ta demande explicite.
