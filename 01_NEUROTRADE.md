# 🧠 NeuroTrade — Guide complet

> Système de trading algorithmique par deep learning avec validation rigoureuse anti-overfitting.

---

## 📖 C'est quoi, en clair ?

NeuroTrade est un programme qui **analyse les données de marché financier** (prix, volumes, indicateurs) et décide automatiquement **quand acheter, vendre ou ne rien faire** sur un actif (action, crypto, forex…).

Ce qui le distingue d'un bot de trading basique :
- Il utilise du **deep learning** (réseau de neurones) pour trouver des patterns complexes.
- Il emploie le **triple-barrier labeling** (technique de Marcos López de Prado) : au lieu de dire bêtement « si ça monte c'est bon », il définit une vraie cible de trade (take profit, stop loss, durée max).
- Il teste ses performances avec le **walk-forward** : il entraîne sur le passé, teste sur le futur réel, avance dans le temps, répète — ce qui évite de « tricher » avec les données futures.
- Il mesure la valeur réelle d'une stratégie avec le **Deflated Sharpe Ratio** qui pénalise les coïncidences statistiques.

---

## 🏗️ Architecture du système

```
Données brutes (OHLCV)
        │
        ▼
  Prétraitement
  (nettoyage, features)
        │
        ▼
Triple-Barrier Labeling
  (définir les trades)
        │
        ▼
   Entraînement
   Bi-LSTM / Transformer
        │
        ▼
Walk-Forward Validation
  (test honnête)
        │
        ▼
Deflated Sharpe Ratio
  (évaluation finale)
        │
        ▼
  Signaux de trading
```

---

## 🚀 Comment l'utiliser

### Installation
```bash
git clone <ton-repo>/neurotrade
cd neurotrade
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Récupérer des données
```bash
# Données gratuites via yfinance
python -m neurotrade.cli fetch --ticker BTC-USD --start 2020-01-01 --end 2024-01-01
```

### Entraîner un modèle
```bash
python -m neurotrade.cli train \
  --ticker BTC-USD \
  --barrier-tp 0.02 \   # take profit +2%
  --barrier-sl 0.01 \   # stop loss -1%
  --barrier-t  5        # durée max 5 jours
```

### Évaluer une stratégie
```bash
python -m neurotrade.cli evaluate --run <run_id>
# → affiche le Deflated Sharpe Ratio, le winrate, le drawdown max
```

### Lancer le backtest complet
```bash
python -m neurotrade.cli backtest --ticker BTC-USD --walk-forward-steps 12
```

---

## 🔧 Comment déboguer

### Le modèle perd de l'argent en backtest
| Symptôme | Cause probable | Fix |
|---|---|---|
| Sharpe négatif | Features pas assez informatives | Ajouter RSI, ATR, volume relatif |
| Sharpe positif en train, négatif en test | Overfitting | Réduire la taille du modèle, ajouter dropout |
| Trop peu de signaux | Seuil de confiance trop élevé | Descendre le threshold de 0.6 à 0.5 |
| Drawdown énorme | Pas de filtre de volatilité | Filtrer les périodes GARCH vol > seuil |

### Débogage data
```bash
# Vérifier les NaN
python -c "import pandas as pd; df=pd.read_parquet('data/BTC-USD.parquet'); print(df.isnull().sum())"

# Vérifier la distribution des labels
python -m neurotrade.cli check-labels --ticker BTC-USD
# → doit être à peu près équilibré : ~33% buy, ~33% sell, ~33% hold
```

### Débogage modèle
```python
# Dans un notebook
from neurotrade.model import load_run
run = load_run("run_20240601")
run.plot_learning_curves()  # loss train vs val → chercher le point de divergence
run.plot_feature_importance()  # quelles features comptent vraiment ?
```

---

## 📈 Comment l'améliorer

### Court terme (semaines)
- [ ] Ajouter des **features de microstructure** : spread bid/ask, imbalance de carnet d'ordres
- [ ] Tester un **Transformer** à la place du Bi-LSTM (souvent meilleur sur les longues séquences)
- [ ] **Ensemble** : combiner 3–5 modèles entraînés différemment, voter sur le signal final

### Moyen terme (mois)
- [ ] **Multi-actifs** : entraîner sur un univers de 20+ actifs en même temps → plus de données, meilleure généralisation
- [ ] **Régime de marché** : détecter bull/bear/sideways et adapter les paramètres (HMM ou clustering)
- [ ] **Gestion du risque dynamique** : Kelly criterion pour dimensionner les positions

### Long terme
- [ ] Connexion à un vrai broker (paper trading d'abord) via **CCXT** (crypto) ou **Interactive Brokers API**
- [ ] Monitoring en temps réel avec dashboard + alertes

---

## 💰 Rendre ça rentable

### Phase 0 — Vérifier que ça marche vraiment
**Condition sine qua non avant tout le reste :**
- Deflated Sharpe Ratio > 1.5 sur au moins 2 ans de walk-forward
- Drawdown max < 20%
- Résultats stables sur plusieurs actifs différents
- Au moins 3 mois de **paper trading** (simulation avec données réelles en direct)

### Phase 1 — Trading pour toi-même
- Ouvre un compte crypto avec un exchange qui a une API (Binance, Kraken…)
- Déploie NeuroTrade en paper trading 3 mois, compare les résultats prédits vs réels
- Si ça tient : commence avec des petits montants (< 500€), scale graduellement

### Phase 2 — Pistes commerciales (si le modèle est vraiment bon)

| Modèle | Comment | Revenus estimés |
|---|---|---|
| **Signaux par abonnement** | Newsletter/bot Discord avec les signaux du jour | 20–100€/mois × abonnés |
| **Copy trading** | Publier ta stratégie sur eToro/Collective2, les autres copient et tu prends une commission | 5–20% des profits des copieurs |
| **Vente du système** | Package + documentation + support → vente one-shot à des traders individuels | 500–2000€/vente |
| **Prop trading** | Passer les évaluations FTMO ou The Funded Trader → trader avec leur capital (jusqu'à 200k€) | 80% des profits |

### ⚠️ Réalité à ne pas oublier
> 95% des systèmes de trading qui marchent en backtest échouent en live. La priorité absolue est la **validation hors-échantillon** rigoureuse. Ne mets pas d'argent réel avant minimum 6 mois de paper trading concluants.

---

## 🛒 Commercialisation complète

### Si tu veux vendre les signaux
1. Crée une page Gumroad ou Substack (gratuit)
2. Publie un **track record transparent** (screenshots de tes trades, P&L mensuel)
3. Commence avec 30 jours d'essai gratuit pour construire la preuve sociale
4. Prix : 29–99€/mois selon la qualité du track record

### Si tu veux vendre le système lui-même
1. Écris une documentation claire + vidéo de démo (Loom)
2. Vends sur Gumroad ou CodeCanyon
3. Propose un support Discord pendant 30 jours post-achat
4. Prix : 299–1499€ en one-shot

### Obligations légales en France
- Les **conseils en investissement** sont une activité réglementée (agrément AMF obligatoire pour vendre des signaux à des tiers)
- L'exception : vendre le **logiciel** (pas les signaux) ne nécessite pas d'agrément
- Pour commencer proprement : vends le code + la méthodologie, pas « les conseils »
- Mentionne toujours que les performances passées ne garantissent pas les résultats futurs

---

## 📊 Métriques à suivre

| Métrique | Bon seuil | Ce qu'elle mesure |
|---|---|---|
| Deflated Sharpe Ratio | > 1.5 | Performance ajustée du risque, anti-chance |
| Max Drawdown | < 20% | Perte max depuis un pic |
| Win Rate | > 50% | % de trades gagnants |
| Profit Factor | > 1.5 | Gains totaux / Pertes totales |
| Calmar Ratio | > 1.0 | Rendement annuel / Drawdown max |

---

## 🔗 Ressources clés

- **Livre de référence** : *Advances in Financial Machine Learning* — Marcos López de Prado (le livre qui définit tout ce que fait NeuroTrade)
- **Données gratuites** : `yfinance`, `ccxt` (crypto), Stooq
- **Paper trading** : Alpaca (US stocks, API gratuite), Binance Testnet (crypto)
- **Évaluation des stratégies** : `quantstats` (pip install quantstats)
