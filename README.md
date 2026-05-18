# Prédiction du prix au m² — Appartements parisiens (DVF)

Proof of Concept ML : prédiction du prix au m² des appartements parisiens à partir des données open data DVF (Demandes de Valeurs Foncières), enrichies avec des données géographiques (transports, espaces verts, équipements INSEE, stationnement).

## Résultats

| Modèle | MAE (€/m²) | RMSE (€/m²) | R² |
|---|---|---|---|
| Régression Linéaire | 2 026 | 2 951 | 0,236 |
| Random Forest | 1 572 | 2 357 | 0,513 |
| Gradient Boosting | 1 825 | 2 690 | 0,365 |
| **Random Forest v2 (log-transform)** | **1 412** | **2 227** | **0,565** |
| Gradient Boosting v2 (log-transform) | 1 826 | 2 739 | 0,342 |

**Meilleur modèle :** Random Forest v2 — MAE = 1 412 €/m² (~14 % du prix médian de 10 300 €/m²)

---

## Installation

```bash
git clone https://github.com/thomasssrss/ml-poc-project.git
cd ml-poc-project
pip install -r requirements.txt
```

---

## Lancer l'application

```bash
python main.py
# ou
python scripts/main.py
```

Ce script :
1. Charge le dataset de test et évalue les modèles configurés dans `src/config.py`
2. Sauvegarde les métriques dans `results/model_metrics.csv`
3. Lance l'application Streamlit sur **http://localhost:8501**

> Si les données ne sont pas présentes, l'évaluation est ignorée et Streamlit se lance directement.

Pour lancer uniquement l'application sans évaluation :

```bash
streamlit run src/app.py
```

---

## Données requises

Les fichiers de données ne sont pas versionnés (taille > 100 Mo). Télécharger et placer dans les dossiers indiqués :

| Fichier | Dossier | Source |
|---|---|---|
| `dvf.csv` | `data/raw/` | [data.gouv.fr — DVF](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/) |
| `emplacement-des-gares-idf.csv` | `data/external/` | [IDFM open data](https://data.iledefrance-mobilites.fr) |
| `espaces_verts.csv` | `data/external/` | [OpenData Paris](https://opendata.paris.fr) |
| `BPE24.csv` | `data/external/` | [INSEE BPE 2024](https://www.insee.fr/fr/statistiques/3568638) |
| `stationnement-voie-publique-emplacements.json` | `data/external/` | [OpenData Paris](https://opendata.paris.fr) |

---

## Reproduire l'entraînement complet

```bash
# 1. Générer les datasets enrichis (EDA + merge géographique)
jupyter notebook notebooks/exploration_data.ipynb
# → data/processed/dvf_paris_enriched.csv

# 2. Feature engineering documenté
jupyter notebook notebooks/feature_engineering.ipynb
# → data/processed/dataset_ml.csv

# 3. Entraîner et comparer les modèles
jupyter notebook notebooks/modelling.ipynb
# → models/linear_regression.pkl
# → models/gradient_boosting.pkl
# → models/gradient_boosting_v2.pkl
# → models/random_forest.pkl        (167 Mo — non versionné)
# → models/random_forest_v2.pkl     (842 Mo — non versionné)
# → results/model_metrics.csv
# → plots/comparaison_modeles.png, predictions_vs_reels.png, feature_importances.png
```

---

## Structure du projet

```
ml-poc-project/
│
├── data/
│   ├── raw/                          # DVF brut (non versionné)
│   ├── external/                     # Datasets géographiques (non versionnés)
│   └── processed/                    # Datasets nettoyés (non versionnés)
│
├── deliverables/                     # Livrables académiques
│   ├── assignment1.md                # Proposition de projet
│   ├── assignment2.md                # Données et features
│   ├── assignment3.md                # Sélection des modèles
│   ├── assignment4.md                # Visualisations
│   └── assignment5.md                # Application Streamlit
│
├── models/                           # Modèles sérialisés
│   ├── linear_regression.pkl         # Baseline (2,7 Ko — versionné)
│   ├── gradient_boosting.pkl         # v1 (1,7 Mo — versionné)
│   ├── gradient_boosting_v2.pkl      # v2 log-transform (3,5 Mo — versionné)
│   ├── random_forest.pkl             # v1 (167 Mo — non versionné)
│   └── random_forest_v2.pkl          # v2 log-transform (842 Mo — non versionné)
│
├── notebooks/
│   ├── exploration_data.ipynb        # EDA + enrichissement géographique
│   ├── feature_engineering.ipynb     # Preprocessing + FE documenté
│   └── modelling.ipynb               # Entraînement + comparaison des modèles
│
├── plots/                            # Visualisations générées par les notebooks
│
├── results/
│   └── model_metrics.csv             # Tableau comparatif des métriques
│
├── scripts/
│   └── main.py                       # Pipeline principal (évaluation + Streamlit)
│
├── src/
│   ├── app.py                        # Application Streamlit (5 pages)
│   ├── config.py                     # Chemins et registre des modèles
│   ├── data.py                       # Chargement, preprocessing, feature engineering
│   ├── metrics.py                    # Calcul MAE, RMSE, R²
│   ├── model_io.py                   # Chargement des fichiers .pkl / .joblib
│   └── results.py                    # Sauvegarde CSV des métriques
│
├── main.py                           # Raccourci : python main.py
├── requirements.txt
└── README.md
```

---

## Application Streamlit

L'application comprend 5 pages accessibles via une barre de navigation horizontale :

| Page | Contenu |
|---|---|
| **Le Projet** | Tableau de bord — données, approche ML, résultats clés |
| **Données brutes** | Prix médian par arrondissement (graphique interactif Plotly) |
| **Feature Engineering** | Log-transform de la cible, features construites, transformations rejetées |
| **Performances** | Comparaison MAE / RMSE / R² des 5 modèles |
| **Estimer un prix** | Estimation interactive par arrondissement ou par adresse exacte |

### Estimation par adresse

Le mode précis géocode l'adresse (API `api-adresse.data.gouv.fr`) et calcule un prix ajusté par :
- **Micro-localisation IDW** : pondération par distance aux 20 centroïdes d'arrondissements
- **Type de voie** : avenue (+5 %), boulevard (+4 %), impasse (−5 %), etc.
- **Proximité Métro/RER** : bonus jusqu'à +2,5 % si station < 200 m
- **Espaces verts** : bonus jusqu'à +3 % si parc > 1 000 m² à moins de 150 m
- **Commerces** : bonus si forte densité dans 500 m
- **Stationnement** : bonus si places disponibles dans 300 m
- **Étage et ascenseur** : bonus/malus selon le marché parisien

---

## Pipeline `scripts/main.py`

```
main()
  ├── _validate_app_entrypoint()     — vérifie que app.py expose build_app()
  ├── _validate_models_config()      — vérifie que config.MODELS est renseigné
  ├── _load_dataset()                — appelle src/data.load_dataset_split()
  ├── _evaluate_models()             — prédit + calcule MAE, RMSE, R² par modèle
  ├── write_metrics()                — sauvegarde results/model_metrics.csv
  └── _launch_streamlit()            — démarre streamlit run src/app.py
```

---

## Modèles versionnés

Les modèles inférieurs à 100 Mo sont inclus dans le dépôt et directement utilisables sans ré-entraînement :

| Fichier | Taille | Algorithme | MAE |
|---|---|---|---|
| `models/linear_regression.pkl` | 2,7 Ko | Pipeline(RobustScaler + LinearRegression) | 2 026 €/m² |
| `models/gradient_boosting.pkl` | 1,7 Mo | HistGradientBoostingRegressor (v1) | 1 825 €/m² |
| `models/gradient_boosting_v2.pkl` | 3,5 Mo | HistGradientBoostingRegressor + log-transform | — |

Les modèles Random Forest (167 Mo et 842 Mo) dépassent la limite GitHub de 100 Mo. Ils sont régénérés en exécutant `notebooks/modelling.ipynb`.
