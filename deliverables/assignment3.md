# Assignment 3 — Sélection et justification des modèles

## Rappel du projet

Prédiction du prix au m² des appartements parisiens à partir des données DVF enrichies avec des données géographiques externes.

---

## 1. Définition du problème ML

Il s'agit d'un problème de **régression supervisée**.

L'objectif est de prédire une valeur numérique continue : le prix au m² (`prix_m2`) d'un appartement parisien, à partir de ses caractéristiques physiques (surface, nombre de pièces, arrondissement) et de sa localisation géographique (distances aux transports, espaces verts, commerces, écoles, etc.).

**Variable cible :** `prix_m2` en €/m² (valeur continue, entre 3 000 et 25 000 €/m²)

**Features en entrée :** 25 variables (voir `deliverables/assignment2.md` pour le détail complet)

**Données disponibles :**
- 111 738 transactions immobilières parisiennes (appartements, 2014–2022)
- Split 80/20 : 89 390 lignes d'entraînement, 22 348 lignes de test

---

## 2. Métrique d'évaluation

Trois métriques sont utilisées, implémentées dans `src/metrics.py` via `compute_metrics(y_true, y_pred)`.

### Métrique principale : MAE (Mean Absolute Error)

```
MAE = moyenne(|prix_réel - prix_prédit|)
```

**Pourquoi c'est la métrique principale ?**
La MAE est exprimée directement en **€/m²**, ce qui la rend immédiatement interprétable dans un contexte business. Un résultat comme "le modèle se trompe en moyenne de 850 €/m²" est compréhensible sans expertise en machine learning.

### Métriques secondaires

| Métrique | Formule | Rôle |
|---|---|---|
| **RMSE** | √(moyenne((réel - prédit)²)) | Pénalise davantage les grosses erreurs ; utile pour détecter si le modèle fait de très mauvaises prédictions ponctuelles |
| **R²** | 1 - SS_res / SS_tot | Mesure la part de variance du prix expliquée par le modèle (0 = nul, 1 = parfait) |

Les trois métriques sont calculées systématiquement pour chaque modèle afin de permettre une comparaison complète.

---

## 3. Protocole d'évaluation

### Split train/test

Le dataset est divisé en deux parties via `sklearn.model_selection.train_test_split` avec `random_state=42` pour la reproductibilité :

- **Train (80 %)** : 89 390 lignes — utilisé pour entraîner les modèles
- **Test (20 %)** : 22 348 lignes — utilisé uniquement pour l'évaluation finale

Le split est effectué de façon aléatoire (pas de stratification sur l'arrondissement) car la distribution des arrondissements est suffisamment équilibrée dans le dataset.

### Pas de data leakage

La colonne `valeur_fonciere` (source du prix_m2) est supprimée avant tout split. Le scaling (RobustScaler) sera fitté uniquement sur le train set et appliqué sur le test set, sans jamais exposer les données de test pendant l'entraînement.

### Évaluation comparative

Les trois modèles sont évalués avec exactement les mêmes métriques (MAE, RMSE, R²) sur le même test set, ce qui garantit une comparaison équitable. Les résultats sont sauvegardés dans `results/model_metrics.csv`.

---

## 4. Les trois modèles sélectionnés

### Modèle 1 — Régression Linéaire

#### Description
La régression linéaire modélise `prix_m2` comme une combinaison linéaire pondérée des 25 features :

```
prix_m2 = w1 × surface + w2 × arrondissement + w3 × distance_station + ... + biais
```

#### Hypothèses principales
- La relation entre chaque feature et le prix est **linéaire et additive**
- Les features sont **indépendantes** entre elles (pas d'interaction)
- Les résidus suivent une distribution normale de variance constante

#### Avantages attendus
- Très rapide à entraîner (quelques secondes)
- Totalement interprétable : chaque coefficient dit combien le prix varie quand la feature augmente d'une unité
- Sert de **baseline** : si un modèle complexe ne fait pas mieux, il y a un problème

#### Limites attendues
- Ne capture pas les relations non-linéaires (ex : l'effet d'une petite surface sur le prix n'est pas proportionnel)
- Ne modélise pas les interactions entre variables (ex : être dans le 6e ET proche d'un parc a un effet combiné)
- Sensible aux outliers et à la multicolinéarité entre features géographiques

#### Adéquation avec le problème
Adéquation **partielle**. Le marché immobilier parisien présente des effets non-linéaires forts (effet arrondissement, micro-localisation) que la régression linéaire ne peut pas capturer. Elle fournit néanmoins une baseline solide et des coefficients interprétables.

---

### Modèle 2 — Random Forest

#### Description
Le Random Forest entraîne un grand nombre d'arbres de décision (typiquement 100 à 500) sur des sous-échantillons aléatoires du dataset, puis fait la moyenne de leurs prédictions. Chaque arbre apprend des règles de segmentation du type :

```
Si arrondissement ≤ 8 ET surface < 35 ET distance_station < 200m → prix_m2 ≈ 13 500 €
```

#### Hypothèses principales
- Les relations entre features et prix peuvent être **non-linéaires** et **non-monotones**
- La combinaison de nombreux modèles faibles produit un modèle robuste (principe du bagging)
- Les features les plus informatives s'expriment via leur **importance relative**

#### Avantages attendus
- Capture les **non-linéarités** et les **interactions** entre variables sans configuration manuelle
- Très robuste aux outliers et aux valeurs aberrantes résiduelles
- Fournit des **feature importances** : on peut identifier quelles variables (arrondissement, surface, distance station) influencent le plus le prix
- Peu sensible aux hyperparamètres de base — fonctionne bien sans tuning poussé

#### Limites attendues
- Moins interprétable que la régression linéaire (boîte noire partielle)
- Peut avoir tendance à **sous-estimer les extrêmes** (très hauts et très bas prix) car il fait une moyenne
- Plus lent à entraîner que la régression linéaire
- Peut surapprendres si les arbres sont trop profonds

#### Adéquation avec le problème
Adéquation **forte**. Le marché immobilier parisien présente exactement les caractéristiques pour lesquelles le Random Forest excelle : relations non-linéaires, interactions entre localisation et caractéristiques du bien, dataset de taille moyenne.

---

### Modèle 3 — Gradient Boosting (XGBoost)

#### Description
Le Gradient Boosting construit les arbres de façon **séquentielle** : chaque nouvel arbre corrige les erreurs du modèle précédent, en se concentrant sur les exemples les plus difficiles à prédire. XGBoost est une implémentation optimisée qui ajoute la régularisation L1/L2 pour éviter le surapprentissage.

```
Modèle final = arbre_1 + arbre_2 (corrige arbre_1) + arbre_3 (corrige arbre_2) + ...
```

#### Hypothèses principales
- Les erreurs résiduelles peuvent être progressivement réduites par des modèles successifs
- La régularisation contrôle la complexité et évite le surapprentissage
- Les relations dans les données sont **complexes et hiérarchiques**

#### Avantages attendus
- Généralement le **modèle le plus performant** sur des données tabulaires structurées comme les nôtres
- Gère nativement les valeurs manquantes
- Propose un **feature importance** par gain (quelle feature réduit le plus l'erreur)
- Régularisation intégrée (paramètres `alpha`, `lambda`) qui limite le surapprentissage

#### Limites attendues
- Plus difficile à interpréter que la régression linéaire
- Nécessite un **tuning d'hyperparamètres** pour atteindre ses meilleures performances (`n_estimators`, `learning_rate`, `max_depth`, etc.)
- Plus lent à entraîner que le Random Forest si mal configuré
- Peut surapprendre si `learning_rate` trop élevé et `n_estimators` trop grand

#### Adéquation avec le problème
Adéquation **très forte**. XGBoost est l'algorithme de référence pour la prédiction de prix immobiliers sur des données tabulaires. La correction itératives des erreurs est particulièrement adaptée aux marchés avec de fortes hétérogénéités locales comme Paris.

---

## 5. Justification du choix des trois modèles

Les trois modèles forment une progression logique en termes de complexité :

| Critère | Régression Linéaire | Random Forest | XGBoost |
|---|---|---|---|
| Complexité | Faible | Moyenne | Élevée |
| Interprétabilité | ✅ Très haute | ⚠️ Moyenne | ⚠️ Moyenne |
| Non-linéarités | ❌ Non | ✅ Oui | ✅ Oui |
| Interactions entre features | ❌ Non | ✅ Oui | ✅ Oui |
| Robustesse aux outliers | ⚠️ Faible | ✅ Forte | ✅ Forte |
| Perf. attendue (MAE) | Référence | Meilleure | Meilleure encore |
| Temps d'entraînement | Très rapide | Rapide | Moyen |

**Pourquoi ces trois-là spécifiquement ?**

1. La **régression linéaire** est indispensable comme baseline. Sans elle, impossible de savoir si les modèles complexes apportent vraiment quelque chose.

2. Le **Random Forest** est robuste, facile à utiliser et fournit des feature importances utiles pour comprendre quelles variables influencent les prix. Il représente un bon compromis performance/interprétabilité.

3. **XGBoost** est le modèle état de l'art pour ce type de problème. Il est systématiquement le plus performant sur des données tabulaires structurées dans les benchmarks publiés.

Cette progression permet aussi de répondre à la question business : si la régression linéaire atteint déjà un MAE satisfaisant, on n'a pas besoin d'un modèle complexe. Si XGBoost fait 30 % mieux, la complexité supplémentaire est justifiée.

---

## 6. Notebooks et reproduction des expériences

### Localisation des notebooks

| Notebook | Rôle |
|---|---|
| `notebooks/exploration_data.ipynb` | Nettoyage DVF + EDA + merge datasets externes → produit `dvf_paris_enriched.csv` |
| `notebooks/feature_engineering.ipynb` | Transformations testées, pipeline documenté → produit `dataset_ml.csv` |
| `notebooks/modelling.ipynb` | Entraînement et comparaison des trois modèles *(à venir)* |

### Comment reproduire les expériences

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Placer les datasets dans les bons dossiers
# data/raw/dvf.csv
# data/external/emplacement-des-gares-idf.csv
# data/external/espaces_verts.csv
# data/external/BPE24.csv

# 3. Exécuter les notebooks dans l'ordre
jupyter notebook notebooks/exploration_data.ipynb     # produit dvf_paris_enriched.csv
jupyter notebook notebooks/feature_engineering.ipynb  # produit dataset_ml.csv
jupyter notebook notebooks/modelling.ipynb            # entraîne et compare les modèles
```

### Chargement des données dans le code

```python
from src.data import load_dataset_split
from src.metrics import compute_metrics

X_train, X_test, y_train, y_test = load_dataset_split()
# → 89 390 lignes d'entraînement, 22 348 de test, 25 features
```
