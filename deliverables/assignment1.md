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

- **Notebook EDA** : `notebooks/exploration_data.ipynb`
- **Dataset brut** : `data/raw/dvf.csv` (non versionné)
- **Dataset nettoyé** : `data/processed/dvf_paris_clean.csv` (non versionné)

---

## Croisement avec les données de transport IDF

### Deuxième dataset : Emplacement des gares IDF

Le dataset `data/external/emplacement-des-gares-idf.csv` est publié en open data par Île-de-France Mobilités (IDFM). Il recense l'ensemble des arrêts et stations de transport en commun d'Île-de-France.

Colonnes utilisées :

| Colonne | Description |
|---|---|
| `Geo Point` | Coordonnées géographiques de la station (latitude, longitude) |
| `nom_long` | Nom complet de la station |
| `res_com` | Réseau / ligne commerciale |
| `indice_lig` | Indice de ligne |
| `mode` | Mode de transport (Métro, RER, Tramway, Bus, Train) |
| `exploitant` | Opérateur (RATP, SNCF, etc.) |

Après nettoyage (suppression des lignes sans coordonnées), le dataset comprend plusieurs centaines de stations couvrant Paris et la petite couronne.

### Méthode de croisement

Pour chaque transaction DVF, les variables d'accessibilité ont été calculées en utilisant `sklearn.neighbors.BallTree` avec la distance haversine (distance sphérique).

Cette approche permet un calcul vectorisé efficace sur l'ensemble du dataset sans boucle Python explicite.

Les coordonnées (latitude, longitude) ont été converties en radians avant d'alimenter le BallTree, conformément aux exigences de la métrique haversine. Les distances en radians sont ensuite converties en mètres via le rayon terrestre moyen (6 371 km).

### Nouvelles features créées

| Feature | Description |
|---|---|
| `distance_station_plus_proche` | Distance en mètres à la station la plus proche |
| `nom_station_plus_proche` | Nom de la station la plus proche |
| `mode_station_plus_proche` | Mode de transport de la station la plus proche |
| `ligne_station_plus_proche` | Numéro de ligne de la station la plus proche |
| `nb_stations_500m` | Nombre de stations dans un rayon de 500 m |
| `nb_stations_1000m` | Nombre de stations dans un rayon de 1 000 m |

### Premiers résultats observés

- La grande majorité des appartements parisiens se trouvent à moins de 500 m d'une station de transport, ce qui reflète la densité du réseau RATP/SNCF à Paris.
- Une légère tendance se dégage : les biens situés à moins de 500 m d'une station présentent un prix m² médian légèrement supérieur à ceux situés au-delà, bien que l'écart reste modeste à l'échelle de Paris.
- La corrélation (Pearson) entre la distance à la station et le prix au m² est faible et négative, confirmant que la proximité aux transports est associée à des prix plus élevés, mais que cet effet seul n'explique qu'une part limitée de la variance.
- Les biens dont la station la plus proche est un RER ou un Tramway présentent des prix m² médians légèrement différents de ceux proches d'une station de Métro.

### Limites du croisement

- **Granularité géographique** : les coordonnées DVF sont souvent approximatives (centroïde de section cadastrale), ce qui peut introduire des erreurs de quelques dizaines de mètres sur la distance calculée.
- **Date des données** : le dataset stations représente l'état actuel du réseau, alors que les transactions DVF couvrent plusieurs années. Des stations ouvertes récemment peuvent biaiser les distances calculées pour les transactions anciennes.
- **Qualité du signal** : à Paris, la densité du réseau de transport est si élevée que la quasi-totalité des biens se trouvent à moins de 500 m d'une station. La variabilité de cette feature est donc plus faible qu'en banlieue.
- **Absence de pondération par fréquence ou ligne** : toutes les stations ont le même poids dans le calcul. Une station de RER Express (forte desserte longue distance) et un arrêt de bus local sont traités identiquement.
- **Dataset brut** : `data/processed/dvf_paris_transport_enriched.csv` (non versionné)
- **Notebook de croisement** : `notebooks/exploration_data.ipynb` (sections 3–6)

## Provenance, collecte et contraintes légales des données

### Dataset principal : DVF

Le dataset principal utilisé est DVF, pour Demandes de valeurs foncières. Il recense les ventes de biens fonciers et immobiliers réalisées sur les dernières années en France. Dans ce projet, il est utilisé pour extraire les transactions d'appartements situées à Paris et calculer une variable cible : le prix au m².

Le fichier utilisé a été téléchargé manuellement depuis les données DVF disponibles en open data. Il est placé localement dans le repository à l'emplacement suivant :

`data/raw/dvf.csv`

Ce fichier brut n'est pas versionné sur GitHub car il est volumineux. Le repository documente donc son emplacement attendu et son utilisation.

### Dataset externe : stations de transport IDF

Un second dataset est utilisé pour enrichir les données immobilières avec une variable d'accessibilité aux transports. Il s'agit du fichier des gares et stations du réseau ferré d'Île-de-France.

Le fichier est placé dans le repository à l'emplacement suivant :

`data/external/emplacement-des-gares-idf.csv`

Ce dataset contient notamment les noms des stations, les modes de transport, les lignes et les coordonnées géographiques. Il permet de calculer des variables comme la distance à la station la plus proche ou le nombre de stations à proximité.

### Méthode de collecte

Les deux datasets ont été obtenus par téléchargement manuel depuis des plateformes open data :

- DVF : données ouvertes des valeurs foncières ;
- Stations IDF : données ouvertes sur les gares et stations du réseau ferré d'Île-de-France.

Aucune méthode de scraping n'a été utilisée. Il n'est donc pas nécessaire de créer un fichier `src/scraping.py`.

### Contraintes légales et usage

Les données DVF sont disponibles en open data, mais elles peuvent contenir des informations sensibles liées aux transactions immobilières. Leur réutilisation ne doit pas permettre la ré-identification de personnes physiques ni l'indexation des données par des moteurs de recherche externes. L'analyse sera donc réalisée à un niveau agrégé et dans un objectif pédagogique, sans tentative d'identification d'individus ou de propriétaires.

Le dataset des stations IDF est utilisé comme donnée géographique ouverte pour enrichir les transactions avec des informations de proximité aux transports.

### Justification du choix

Le choix de DVF est pertinent car il fournit des prix de vente réellement observés sur le marché immobilier, contrairement à des prix d'annonces qui peuvent être biaisés. Le choix du dataset des stations IDF est pertinent car la proximité aux transports est un facteur susceptible d'influencer le prix au m² des appartements parisiens.

Le croisement des deux sources permet donc de construire un dataset plus riche pour un modèle de régression supervisée.