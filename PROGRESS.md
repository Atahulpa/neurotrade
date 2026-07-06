# PROGRESS — NeuroTrade

Dernière mise à jour : 2026-06-07

## Légende
- ✅ Validé (commit effectué)
- 🔄 En cours
- ⬜ À venir

---

## Phase 0 — Échafaudage

| Étape | Description | Statut | Commit |
|-------|-------------|--------|--------|
| 0 | Repo, env uv, pyproject.toml, squelette archi, outillage qualité, CLAUDE.md, README, PROGRESS.md | ✅ `make check` vert (ruff + mypy strict + 12 tests) | `f0fa411` |

---

## Phase 1 — Recherche / Backtest offline

| Étape | Couche | Description | DoD | Statut | Commit |
|-------|--------|-------------|-----|--------|--------|
| 1 | `data/` | Fetch OHLCV + cache parquet + gestion trous + resampling | Série propre rechargée depuis cache ; tests verts | ✅ (13 tests) | `234cd4e` |
| 2 | `features/` | < 10 features, pipeline, scaler train-only | Test anti-fuite scaler vert | ✅ (19 tests) | `234cd4e` |
| 3 | `labeling/` | Triple-barrier TP/SL/horizon | Labels corrects sur 3 cas connus | ✅ (8 tests) | `234cd4e` |
| 4 | `models/` | MLP + 1D-CNN + LSTM, interface commune, early stopping | Entraînement déterministe ; courbes de perte | ✅ (7 tests) | `a70bcca` |
| 5 | `backtest/` walk-forward | Fenêtres glissantes + purge + embargo | Test anti-fuite vert | ✅ (15 tests) | `072c394` |
| 6 | `backtest/` exécution | Frais, slippage, funding, latence | PnL recalculé à la main concorde | ✅ (7 tests) | `072c394` |
| 7 | `backtest/` risque | Sizing, stop journalier, kill-switch | Déclenchements testés | ✅ (14 tests) | `072c394` |
| 8 | `evaluation/` | Sharpe, DSR, PBO, drawdown, turnover, rapport | Un rapport par run ; métriques vérifiées | ✅ (21 tests) | voir ci-dessous |
| 9 | `experiments/` | Orchestrateur complet data→features→labels→model→backtest→eval | Pipeline intégré, `neurotrade backtest` opérationnel | ✅ mypy+ruff | voir ci-dessous |

---

## Phase 2 — Live / Paper trading

> **Non démarrée.** Démarrage uniquement sur demande explicite après validation Phase 1.

---

## Décisions techniques en suspens

- Choix de la source de données : `ccxt` (crypto) ou `yfinance` (actions) — à décider en étape 1
- Marché de départ : BTC/USDT 1 min (forte liquidité, frais connus) — à confirmer
- Architecture prioritaire : MLP baseline d'abord, CNN/LSTM en étape 4

---

## Notes

- Seeds fixés : `numpy`, `random`, `torch` (voir `config/schema.py`)
- Tout l'output est structuré via `logging`, jamais de `print`
- Cache parquet dans `data/` (ignoré par git)
- Rapports dans `reports/` (ignorés par git)
