# Assignment 2 — Préparation des données et feature engineering

## Rappel du projet

Prédiction du prix au m² des appartements parisiens à partir des données DVF (Demandes de Valeurs Foncières), enrichies avec des données géographiques externes.

Problème : **régression supervisée**
Variable cible : `prix_m2 = valeur_foncière / surface_reelle_bati`

---

## 1. Étapes de nettoyage des données

Le nettoyage est réalisé dans `notebooks/exploration_data.ipynb` et implémenté dans `src/data.py`.

### 1.1 Filtrage initial du DVF brut

Le fichier DVF brut contient 420 066 lignes couvrant toutes les mutations immobilières de France. Les filtres appliqués sont :

| Filtre | Lignes restantes |
|---|---|
| Fichier brut | 420 066 |
| Nature mutation = "Vente" | ~250 000 |
| Code département = 75 (Paris) | ~200 000 |
| Type local = "Appartement" | ~190 000 |
| Surface réelle bâtie > 0 | 188 426 |
| Prix au m² entre 3 000 et 25 000 € | **156 077** |

**Justification des seuils outliers (3 000–25 000 €/m²) :** les valeurs en dehors de cette plage correspondent à des erreurs de saisie, des mutations complexes (plusieurs lots agrégés) ou des biens atypiques. Ces 32 349 lignes supprimées (~17 %) dégradent l'apprentissage sans apporter d'information utile.

### 1.2 Suppression des colonnes inutiles

Les colonnes suivantes sont supprimées avant modélisation (liste dans `src/data.py`, constante `COLS_TO_DROP`) :

| Catégorie | Colonnes supprimées | Raison |
|---|---|---|
| Identifiants | `id_mutation`, `id_parcelle`, `numero_volume`, etc. | Aucune valeur prédictive |
| Fuite de données | `valeur_fonciere` | `prix_m2` en est dérivé directement |
| Adresse précise | `adresse_nom_voie`, `adresse_code_voie`, etc. | Cardinalité trop élevée |
| Lots Carrez | `lot1_*` à `lot5_*`, `nombre_lots` | Quasi vides pour les appartements |
| Constantes post-filtrage | `nature_mutation`, `type_local`, `code_type_local` | Une seule valeur après filtrage |
| Redondants avec code postal | `code_commune`, `nom_commune`, `code_departement` | Tous Paris 75 |
| Quasi vides | `code_nature_culture`, `surface_terrain` | > 95 % de valeurs manquantes |

### 1.3 Gestion des valeurs manquantes

| Colonne | Traitement | Justification |
|---|---|---|
| `nombre_pieces_principales` | Imputation par la médiane | Variable numérique avec peu de manquants, médiane robuste aux outliers |
| `latitude` / `longitude` | Suppression des lignes sans coordonnées | Indispensables pour le merge géographique, très peu de lignes concernées |
| `code_postal` | Suppression des lignes sans code postal | Nécessaire pour extraire l'arrondissement |

---

## 2. Transformations appliquées

Toutes les transformations sont documentées et testées dans `notebooks/feature_engineering.ipynb`, puis implémentées dans `src/data.py`.

### 2.1 Extraction des features temporelles

**Transformation :** extraction de l'année et du mois depuis `date_mutation`.

```python
df["annee"] = df["date_mutation"].dt.year
df["mois"]  = df["date_mutation"].dt.month
df = df.drop(columns=["date_mutation"])
```

**Justification :** la date brute n'est pas exploitable directement par les modèles. L'année capture la tendance temporelle du marché immobilier (hausse continue des prix parisiens). Le mois capture la saisonnalité des transactions.

### 2.2 Extraction de l'arrondissement

**Transformation :** extraction du numéro d'arrondissement (1–20) depuis le code postal.

```python
df["arrondissement"] = (
    df["code_postal"].astype(float).astype(int).astype(str).str[-2:].astype(int)
)
df = df.drop(columns=["code_postal"])
```

**Justification :** le code postal complet (ex. 75006) est redondant. L'arrondissement seul (6) est suffisant et plus compact. C'est une des variables les plus prédictives : le prix médian varie de 8 400 €/m² (19e) à 14 600 €/m² (6e).

### 2.3 Encodage One-Hot du mode de transport

**Transformation :** encodage OHE de la variable catégorielle `mode_station_plus_proche` (Métro, RER, Tramway, etc.).

```python
mode_dummies = pd.get_dummies(df["mode_station_plus_proche"], prefix="mode", drop_first=True)
df = pd.concat([df, mode_dummies], axis=1)
df = df.drop(columns=["mode_station_plus_proche"])
```

**Justification :** les modèles de machine learning ne peuvent pas traiter directement des chaînes de caractères. L'OHE crée une colonne binaire par modalité. `drop_first=True` évite la multicolinéarité parfaite.

---

## 3. Nouvelles features créées

### 3.1 Features dérivées du DVF

| Feature | Source | Description |
|---|---|---|
| `annee` | `date_mutation` | Année de la transaction |
| `mois` | `date_mutation` | Mois de la transaction |
| `arrondissement` | `code_postal` | Numéro d'arrondissement (1–20) |

### 3.2 Features géographiques (datasets externes)

Toutes les features géographiques sont calculées via **BallTree avec la métrique haversine** (sklearn), ce qui permet un calcul vectorisé de distances sphériques sans boucle Python.

**Stations de transport IDF (IDFM open data) :**

| Feature | Description |
|---|---|
| `distance_station_plus_proche` | Distance en mètres à la station la plus proche |
| `nb_stations_500m` | Nombre de stations dans un rayon de 500 m |
| `nb_stations_1000m` | Nombre de stations dans un rayon de 1 000 m |
| `mode_Metro` / `mode_RER` / ... | Mode de transport de la station la plus proche (OHE) |

**Espaces verts de Paris (OpenData Paris) :**

| Feature | Description |
|---|---|
| `distance_ev_plus_proche` | Distance en mètres à l'espace vert le plus proche |
| `surface_ev_plus_proche` | Surface en m² de l'espace vert le plus proche |
| `nb_ev_500m` | Nombre d'espaces verts dans un rayon de 500 m |

**Équipements INSEE BPE 2024 (INSEE open data) :**

| Feature | Description |
|---|---|
| `distance_education_plus_proche` | Distance en mètres à l'école la plus proche |
| `nb_education_500m` | Nombre d'écoles dans un rayon de 500 m |
| `distance_sante_plus_proche` | Distance en mètres au soin de santé le plus proche |
| `nb_sante_500m` | Nombre d'équipements de santé dans un rayon de 500 m |
| `distance_commerce_plus_proche` | Distance en mètres au commerce le plus proche |
| `nb_commerce_500m` | Nombre de commerces dans un rayon de 500 m |
| `distance_loisirs_plus_proche` | Distance en mètres au lieu de loisirs le plus proche |
| `nb_loisirs_500m` | Nombre de lieux de loisirs dans un rayon de 500 m |

**Total : 25 features en entrée du modèle.**

---

## 4. Justification des choix effectués

### Pourquoi garder latitude et longitude ?

Les coordonnées GPS sont gardées comme features de localisation brute en complément de l'arrondissement. Elles permettent aux modèles non-linéaires (Random Forest, XGBoost) de capturer des effets de micro-localisation intra-arrondissement que le seul numéro d'arrondissement ne peut pas encoder.

### Pourquoi utiliser RobustScaler plutôt que StandardScaler ?

Le dataset contient des outliers résiduels (prix au m² jusqu'à 25 000 €, surfaces variables). RobustScaler utilise la médiane et l'IQR (interquartile range) au lieu de la moyenne et l'écart-type, ce qui le rend insensible aux valeurs extrêmes. Le scaling n'est pas appliqué dans `data.py` (il sera intégré dans le pipeline sklearn lors de la modélisation).

### Pourquoi ne pas appliquer le log sur prix_m2 ?

Le log-transform améliore la normalité de la distribution de la cible et peut aider les modèles linéaires. Cependant, cela complique l'interprétation des métriques (MAE en log-€/m² plutôt qu'en €/m²). Ce choix est reporté à la phase de modélisation pour le tester sur chaque modèle séparément.

---

## 5. Alternatives testées et non retenues

Toutes les alternatives sont testées dans `notebooks/feature_engineering.ipynb`.

| Transformation | Testée | Raison du rejet |
|---|---|---|
| **Encodage ordinal de l'arrondissement** | ✅ Oui | L'arrondissement est déjà un entier (1–20), l'encodage ordinal n'apporte rien de plus |
| **StandardScaler** | ✅ Oui | Sensible aux outliers ; RobustScaler préféré |
| **Log-transform sur prix_m2** | ✅ Oui | Normalise la distribution mais complique l'interprétation des métriques en €/m² ; reporté à la modélisation |
| **PCA (Analyse en Composantes Principales)** | ✅ Oui | Réduit la dimensionnalité mais détruit l'interprétabilité des features. Avec seulement 25 features, la réduction n'est pas nécessaire et nuit à la lisibilité des feature importances |

---

## 6. Impact attendu des transformations sur les modèles

| Transformation | Impact attendu |
|---|---|
| Suppression des colonnes inutiles | Réduit le bruit et le risque de surapprentissage |
| Extraction annee/mois | Permet aux modèles de capturer les tendances temporelles du marché |
| Extraction arrondissement | Variable très prédictive : forte corrélation avec prix_m2 (écart de ~6 000 €/m² entre 6e et 19e) |
| OHE mode transport | Permet aux modèles linéaires d'exploiter le type de transport le plus proche |
| Features géographiques (distances + densités) | Enrichissement majeur : capturent l'attractivité du quartier au-delà de l'arrondissement |
| Imputation par médiane | Évite la perte de lignes pour une variable peu manquante |

---

## 7. Métriques d'évaluation

Les métriques sont implémentées dans `src/metrics.py` via la fonction `compute_metrics(y_true, y_pred)`.

| Métrique | Description | Interprétation |
|---|---|---|
| **MAE** | Erreur absolue moyenne | Erreur moyenne du modèle en €/m² — directement interprétable business |
| **RMSE** | Racine de l'erreur quadratique moyenne | Pénalise davantage les grandes erreurs ; en €/m² |
| **R²** | Coefficient de détermination | Part de variance expliquée (1.0 = parfait, 0.0 = modèle nul) |

**Exemple de lecture :** si MAE = 800 €/m², le modèle se trompe en moyenne de 800 €/m², soit ~8 % d'erreur sur un prix médian de 10 300 €/m².

---

## 8. Datasets et notebooks

### Où sont stockés les datasets

Tous les datasets sont stockés **localement uniquement** et exclus de GitHub via `.gitignore` (`data/*` ignoré, seuls les `.gitkeep` sont versionnés).

| Dataset | Emplacement local | Source | Sur GitHub |
|---|---|---|---|
| DVF brut | `data/raw/dvf.csv` | data.gouv.fr | ❌ Non |
| DVF nettoyé | `data/processed/dvf_paris_clean.csv` | Produit par le notebook | ❌ Non |
| DVF enrichi | `data/processed/dvf_paris_enriched.csv` | Produit par le notebook | ❌ Non |
| Dataset ML final | `data/processed/dataset_ml.csv` | Produit par le notebook | ❌ Non |
| Stations IDF | `data/external/emplacement-des-gares-idf.csv` | IDFM open data | ❌ Non |
| Espaces verts | `data/external/espaces_verts.csv` | OpenData Paris | ❌ Non |
| BPE 2024 | `data/external/BPE24.csv` | INSEE open data | ❌ Non |

### Comment obtenir les datasets transformés

1. Télécharger les datasets sources (DVF, IDFM, espaces verts, BPE) depuis leurs sources open data respectives et les placer dans les dossiers `data/raw/` et `data/external/`
2. Exécuter le notebook `notebooks/exploration_data.ipynb` en entier — il produit `data/processed/dvf_paris_enriched.csv`
3. Exécuter le notebook `notebooks/feature_engineering.ipynb` — il documente toutes les transformations et produit `data/processed/dataset_ml.csv`

### Comment charger les données pour la modélisation

```python
from src.data import load_dataset_split

X_train, X_test, y_train, y_test = load_dataset_split()
# X_train : 89 390 lignes × 25 features
# X_test  : 22 348 lignes × 25 features
```

La fonction `load_dataset_split()` applique automatiquement tout le pipeline de nettoyage et de feature engineering. Elle charge `dvf_paris_enriched.csv` si disponible, sinon `dvf_paris_clean.csv` (fallback sans features géographiques).
