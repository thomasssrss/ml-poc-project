# Assignment 1 — Proposition de projet ML

## Sujet choisi

Prédiction du prix au m² des appartements à Paris à partir des données DVF.

## Type de problème ML

Le projet correspond à un problème de régression supervisée.

L'objectif est de prédire une valeur numérique continue : le prix au m² d'un bien immobilier parisien à partir de ses caractéristiques.

## Objectif business

L'objectif business est d'aider un acteur immobilier à estimer plus rapidement la valeur d'un appartement à Paris.

Un tel modèle pourrait être utile pour :

- aider une agence immobilière à proposer une première estimation de prix ;
- aider un vendeur à positionner son bien au bon prix ;
- aider un acheteur à repérer un bien potentiellement surévalué ou sous-évalué ;
- comparer les prix selon les arrondissements ou les caractéristiques du bien ;
- mieux comprendre les facteurs qui influencent les prix immobiliers à Paris.

## Dataset envisagé

Je souhaite utiliser le dataset DVF, pour "Demandes de valeurs foncières".

Ce dataset est publié en open data et recense les mutations immobilières réalisées en France. Il contient notamment des informations sur :

- la date de mutation ;
- la valeur foncière, c'est-à-dire le prix de vente ;
- la nature de la mutation ;
- le type de local ;
- la surface réelle bâtie ;
- le nombre de pièces principales ;
- la commune ;
- le code postal ;
- la section cadastrale ;
- la surface du terrain lorsqu'elle est disponible.

Pour ce projet, je souhaite filtrer les données afin de travailler uniquement sur les appartements situés à Paris.

## Variable cible

La variable cible sera le prix au m².

Elle pourra être calculée de la manière suivante :

prix au m² = valeur foncière / surface réelle bâtie

L'objectif du modèle sera donc de prédire ce prix au m² à partir des caractéristiques du bien.

## Variables explicatives possibles

Les variables utilisées pour entraîner le modèle pourraient être :

- arrondissement ou code postal ;
- surface réelle bâtie ;
- nombre de pièces principales ;
- type de local ;
- date de vente ;
- année de vente ;
- mois de vente ;
- localisation approximative ;
- surface du terrain si disponible ;
- éventuellement des variables créées comme le prix moyen par arrondissement.

## Contexte ML

Le marché immobilier parisien présente de fortes différences selon les arrondissements, la surface du bien, le nombre de pièces et la période de vente.

L'objectif du modèle est d'apprendre ces relations à partir des transactions passées pour estimer le prix au m² d'un bien.

Ce projet permet aussi de relier un problème business concret à un modèle de machine learning : l'estimation immobilière.

## Modèles envisageables

Je pourrais commencer avec plusieurs modèles de régression :

- régression linéaire ;
- arbre de décision ;
- random forest ;
- gradient boosting.

L'objectif sera de comparer un modèle simple avec des modèles plus avancés.

## Métriques d'évaluation

Pour évaluer les modèles, je pourrais utiliser :

- MAE : erreur absolue moyenne ;
- RMSE : racine de l'erreur quadratique moyenne ;
- R² : capacité du modèle à expliquer la variance du prix.

La MAE sera particulièrement intéressante, car elle permettra d'interpréter l'erreur moyenne du modèle en euros par m².

## Objectif final du POC

L'objectif final est de construire un proof of concept capable de prédire le prix au m² d'un appartement parisien à partir de données DVF.

Le projet devra montrer :

- un nettoyage des données DVF ;
- un filtrage sur Paris et les appartements ;
- une analyse exploratoire des prix par arrondissement ;
- un entraînement de plusieurs modèles de régression ;
- une comparaison des performances ;
- une visualisation des résultats ;
- une conclusion business sur l'utilité et les limites du modèle.


## Première idée de filtrage des données

Pour limiter le périmètre du projet, je commencerai par filtrer uniquement les ventes correspondant à des appartements à Paris. Je pourrai ensuite supprimer les lignes avec des surfaces manquantes ou incohérentes, puis calculer le prix au m² pour chaque transaction.

---

## Premières analyses exploratoires EDA

### Chiffres clés

| Indicateur | Valeur |
|---|---|
| Lignes dans le fichier brut | 420 066 |
| Lignes après filtrage (Vente / Paris / Appartement / surface > 0) | 188 426 |
| Lignes après suppression des outliers (prix_m2 entre 3 000 et 25 000 €) | 156 077 |
| Prix médian au m² | **10 300 €/m²** |
| Prix moyen au m² | 10 771 €/m² |

### Arrondissements les plus chers (prix m² médian)

| Arrondissement | Prix m² médian |
|---|---|
| 75006 (Saint-Germain-des-Prés) | 14 646 €/m² |
| 75007 (Invalides / Tour Eiffel) | 14 483 €/m² |
| 75004 (Marais / Île de la Cité) | 12 933 €/m² |
| 75001 (Louvre / Les Halles) | 12 700 €/m² |
| 75008 (Champs-Élysées) | 12 304 €/m² |

### Arrondissements les moins chers (prix m² médian)

| Arrondissement | Prix m² médian |
|---|---|
| 75012 (Nation / Vincennes) | 9 714 €/m² |
| 75018 (Montmartre / La Chapelle) | 9 321 €/m² |
| 75013 (Gobelins / Chinatown) | 9 286 €/m² |
| 75020 (Belleville / Ménilmontant) | 8 670 €/m² |
| 75019 (La Villette / Buttes-Chaumont) | 8 409 €/m² |

### Limites observées dans les données

- Les colonnes `valeur_fonciere`, `surface_reelle_bati` et `code_departement` sont de type `object` dans le CSV brut et nécessitent une conversion explicite.
- Une même mutation (id_mutation) peut apparaître sur plusieurs lignes (un lot par ligne), ce qui peut introduire des doublons si on ne filtre pas correctement par `type_local`.
- Les valeurs extrêmes de prix_m2 (< 3 000 ou > 25 000 €/m²) représentent environ 17 % des observations filtrées (32 349 lignes supprimées). Elles correspondent probablement à des erreurs de saisie, des biens atypiques ou des mutations complexes.
- Peu de variables explicatives directement exploitables : pas d'informations sur l'étage, l'état du bien, ou la proximité aux transports.
- La surface Carrez (surface des lots) est souvent manquante ; la variable `surface_reelle_bati` est utilisée à la place.

### Localisation des fichiers

- **Notebook EDA** : `notebooks/01_dvf_paris_eda.ipynb`
- **Dataset brut** : `data/raw/dvf.csv` (non versionné)
- **Dataset nettoyé** : `data/processed/dvf_paris_clean.csv` (non versionné)