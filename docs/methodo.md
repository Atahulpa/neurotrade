# Méthodologie — NeuroTrade

## Walk-forward glissant

Au lieu d'entraîner une fois sur toutes les données et de backtester (biais de look-ahead),
le walk-forward divise le temps en fenêtres glissantes :

```
|←── train (N bougies) ──→|← val →|← embargo →|← test (M bougies) →|
                                                ↑
                            Le modèle "trade" ici pendant que le suivant s'entraîne
```

- **Cadence de ré-entraînement** : toutes les `step_bars` bougies (ex. 1 440 = 1 jour).
- **Fenêtre d'entraînement** : `train_window_bars` bougies (ex. 21 600 ≈ 15 jours).
- **Pipeline décalé** : pendant qu'un modèle simule des trades sur la fenêtre courante,
  le suivant s'entraîne sur la fenêtre décalée d'un pas — pas de temps mort.

## Purge et embargo

Sans purge, les features calculées sur une bougie de train peuvent "contaminer" les labels
de validation si les deux se chevauchent (ex. label construit sur horizon 60 bougies).

- **Purge** : retire de val toutes les observations dont l'horizon de label chevauche la
  fin de train. Doit être ≥ `horizon_bars`.
- **Embargo** : retire les premières `embargo_bars` bougies de test après la fin de val,
  pour éviter que le délai d'exécution réel crée une fuite indirecte.

Référence : López de Prado, *Advances in Financial Machine Learning*, chap. 7.

## Labeling triple-barrier

Pour chaque bougie *t*, on cherche lequel des trois événements se produit en premier :

1. **Take-profit (TP)** : prix monte de `tp_pct` → label **+1**
2. **Stop-loss (SL)** : prix baisse de `sl_pct` → label **−1**
3. **Horizon** : ni TP ni SL dans les `horizon_bars` bougies → label **0** (neutre)

Avantage sur les rendements bruts : le label est lié à un événement de marché réel,
pas à une fenêtre de temps arbitraire. Cela réduit le bruit dans la cible.

## Deflated Sharpe Ratio (DSR)

Le Sharpe ratio brut est biaisé quand on teste de nombreuses stratégies (sélection
multiple). Le DSR corrige ce biais en pénalisant la probabilité que le Sharpe observé
soit dû au hasard :

```
DSR = Φ[ (SR - SR*) × √(T-1) / √(1 - γ₃·SR + γ₄·SR²/4) ]
```

où `SR*` est le Sharpe attendu sous H₀, `γ₃` et `γ₄` sont les moments de la distribution
des rendements. Un DSR > 0.95 indique un edge statistiquement robuste.

Référence : Bailey & López de Prado, "The Deflated Sharpe Ratio", 2014.

## Probability of Backtest Overfitting (PBO)

Le PBO mesure la probabilité que la meilleure stratégie en in-sample soit la pire en
out-of-sample, via combinatorial symmetric cross-validation (CSCV) :

- On divise la série en S blocs de temps.
- On exhaustivement teste toutes les combinaisons IS/OOS possibles.
- PBO = fréquence où le rang IS et le rang OOS sont inversés.

Un PBO proche de 0.5 signifie que la sélection est essentiellement aléatoire.
On vise PBO < 0.1 pour avoir confiance dans la sélection de stratégie.

Référence : Bailey et al., "The Probability of Backtest Overfitting", 2015.

## Simulateur d'exécution

Pour qu'un backtest soit honnête, chaque trade doit intégrer :

- **Frais maker/taker** : selon si l'ordre est dans le carnet ou au marché.
- **Slippage paramétrique** : l'impact de marché d'un ordre dépend de la taille.
  Ici, modèle linéaire simple : `slippage = slippage_bps × taille_relative`.
- **Funding rate** : pour les contrats perpétuels crypto, paiement toutes les 8 h.
- **Latence** : le trade est exécuté sur la bougie suivante (`latency_bars`), jamais
  sur la bougie courante (sinon look-ahead biais).

## Seuil d'edge net de coûts

On ne trade que si l'edge prédit par le modèle dépasse les coûts totaux plus une marge :

```
edge_prédit > fee_maker + slippage_bps/10000 + min_ret_to_trade
```

Cela minimise le nombre de transactions par construction, ce qui protège contre le
drag des frais sur des positions marginales.

## Kill-switch

Si la volatilité ou le spread dépassent des seuils paramétriques (z-score > 3σ),
le système se met flat (ferme toutes les positions) et se met en pause.
Cela protège contre les chocs externes (annonces macro, pannes d'exchange, flash crashes).
