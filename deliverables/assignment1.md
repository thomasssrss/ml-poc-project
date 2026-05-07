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

## Croisement avec les données externes

Trois datasets externes ont été croisés avec le DVF pour enrichir les features. La méthode utilisée est identique pour les trois : `sklearn.neighbors.BallTree` avec la métrique haversine, permettant un calcul vectorisé de distances sphériques sans boucle Python. Les coordonnées sont converties en radians avant d'alimenter le BallTree, puis reconverties en mètres via le rayon terrestre moyen (6 371 km).

---

### Deuxième dataset : Stations de transport IDF

Le dataset `data/external/emplacement-des-gares-idf.csv` est publié en open data par Île-de-France Mobilités (IDFM). Il recense l'ensemble des arrêts et stations de transport en commun d'Île-de-France (Métro, RER, Tramway, Bus, Train).

**Features créées :**

| Feature | Description |
|---|---|
| `distance_station_plus_proche` | Distance en mètres à la station la plus proche |
| `nom_station_plus_proche` | Nom de la station la plus proche |
| `mode_station_plus_proche` | Mode de transport de la station la plus proche |
| `ligne_station_plus_proche` | Numéro de ligne de la station la plus proche |
| `nb_stations_500m` | Nombre de stations dans un rayon de 500 m |
| `nb_stations_1000m` | Nombre de stations dans un rayon de 1 000 m |

**Premiers résultats :** La quasi-totalité des appartements parisiens se trouvent à moins de 500 m d'une station, ce qui reflète la densité du réseau RATP/SNCF. La corrélation (Pearson) entre la distance à la station et le prix au m² est faible et négative : la proximité aux transports est associée à des prix légèrement plus élevés, mais cet effet seul n'explique qu'une part limitée de la variance.

**Limites :** Les coordonnées DVF sont approximatives (centroïde cadastral). Le dataset stations représente l'état actuel du réseau, pas celui au moment des transactions.

---

### Troisième dataset : Espaces verts de Paris

Le dataset `data/external/espaces_verts.csv` est publié en open data par la Ville de Paris. Il recense 2 528 espaces verts (parcs, squares, jardinières) avec leurs coordonnées géographiques et leur surface.

**Features créées :**

| Feature | Description |
|---|---|
| `distance_ev_plus_proche` | Distance en mètres à l'espace vert le plus proche |
| `surface_ev_plus_proche` | Surface en m² de l'espace vert le plus proche |
| `nb_ev_500m` | Nombre d'espaces verts dans un rayon de 500 m |

**Premiers résultats :** Une légère corrélation négative entre la distance à un espace vert et le prix au m² est observée, confirmant que la proximité à un parc est associée à des prix plus élevés.

**Limites :** Le dataset inclut toutes les typologies d'espaces verts, y compris de simples jardinières de voirie, ce qui peut diluer l'effet des grands parcs.

---

### Quatrième dataset : Base Permanente des Équipements INSEE 2024 (BPE)

Le dataset `data/external/BPE24.csv` est publié par l'INSEE. Il recense l'ensemble des équipements et services disponibles sur le territoire français (2,8 millions de lignes au national). Seuls les équipements parisiens susceptibles d'influencer le prix au m² ont été conservés :

| Catégorie | Codes INSEE | Équipements |
|---|---|---|
| Éducation | A501, A502, A503 | Écoles maternelles, élémentaires, primaires |
| Santé | D265, D307, D269 | Médecins généralistes, pharmacies, infirmiers |
| Commerces | B201, B302, B310, B311 | Boulangeries, banques, restaurants, cafés |
| Loisirs | F101, F111 | Cinémas, bibliothèques |

**Features créées (par catégorie) :**

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

**Limites :** La BPE 2024 représente l'état actuel des équipements, pas celui au moment des transactions DVF. Certains équipements peuvent avoir ouvert ou fermé depuis.

---

## Provenance, collecte et contraintes légales des données

### Localisation des fichiers

Tous les datasets sont stockés localement et ne sont pas versionnés sur GitHub (`.gitignore` couvre `data/`). Le repository documente leur emplacement attendu.

| Dataset | Emplacement local | Source |
|---|---|---|
| DVF | `data/raw/dvf.csv` | data.gouv.fr — DVF open data |
| Stations IDF | `data/external/emplacement-des-gares-idf.csv` | IDFM open data |
| Espaces verts | `data/external/espaces_verts.csv` | OpenData Paris |
| BPE 2024 | `data/external/BPE24.csv` | INSEE open data |
| Stationnement | `data/external/stationnement-voie-publique-emplacements.json` | OpenData Paris |
| Dataset enrichi | `data/processed/dvf_paris_enriched.csv` | Produit par le notebook |

### Méthode de collecte

Tous les datasets ont été obtenus par téléchargement manuel depuis des plateformes open data. Aucun scraping n'a été utilisé.

### Comment exécuter le notebook

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le notebook
jupyter notebook notebooks/exploration_data.ipynb
```

Le notebook `notebooks/exploration_data.ipynb` réalise en séquence : chargement DVF → EDA → merge transports → merge espaces verts → merge BPE → sauvegarde du dataset enrichi dans `data/processed/dvf_paris_enriched.csv`.

Le dataset stationnement est traité directement dans `notebooks/modelling.ipynb` (section 2b) via BallTree haversine, sans nécessiter de re-générer le dataset enrichi.

### Contraintes légales et usage

Les données DVF contiennent des informations sur des transactions immobilières réelles. Leur réutilisation est limitée à un usage agrégé et pédagogique, sans tentative de ré-identification de personnes. Les quatre datasets externes (IDFM, OpenData Paris ×2, INSEE) sont en licence ouverte et librement réutilisables.

### Justification du choix des datasets

- **DVF** : prix de vente réellement observés (non des prix d'annonces biaisés)
- **Stations IDF** : la proximité aux transports est un facteur classique de valorisation immobilière
- **Espaces verts** : la présence de parcs est connue pour avoir un effet positif sur les prix de l'immobilier résidentiel
- **BPE** : la densité en services (écoles, commerces, santé) reflète l'attractivité d'un quartier et influence directement les prix au m²
- **Stationnement** : à Paris où le stationnement est rare et coûteux, la proximité à des places de stationnement voiture est un facteur de confort pour les résidents et peut influencer la valeur perçue d'un bien