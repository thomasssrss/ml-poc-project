# Assignment 5 — Application Streamlit

## Rappel du projet

Prédiction du prix au m² des appartements parisiens à partir des données DVF enrichies avec des données géographiques externes. L'application Streamlit constitue l'interface finale qui permet de démontrer l'ensemble du travail réalisé : exploration des données, feature engineering, comparaison des modèles, et estimation interactive.

---

## Lancement de l'application

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'application
streamlit run src/app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse `http://localhost:8501`.

Les datasets géographiques suivants doivent être présents dans `data/external/` pour que la page d'estimation par adresse soit pleinement fonctionnelle :
- `emplacement-des-gares-idf.csv` — stations Métro/RER
- `espaces_verts.csv` — espaces verts parisiens
- `BPE24.csv` — équipements INSEE 2024
- `stationnement-voie-publique-emplacements.json` — stationnement voie publique

Les pages "Le Projet", "Données brutes", "Feature Engineering", "Performances" et l'onglet "Par arrondissement" fonctionnent sans ces fichiers.

---

## Objectif de l'interface

L'application remplit trois objectifs complémentaires :

1. **Démonstration** — présenter le projet, les données, l'approche ML et les résultats à un interlocuteur non-technique en quelques clics.
2. **Exploration** — permettre d'explorer interactivement les données brutes (prix par arrondissement), les transformations appliquées (log-transform, features construites) et les performances comparées des cinq modèles.
3. **Prédiction** — estimer le prix au m² d'un appartement en saisissant ses caractéristiques, soit par arrondissement (estimation rapide), soit par adresse exacte (estimation affinée par micro-localisation et données géographiques temps réel).

---

## Structure de l'application

### Fichier principal

```
src/app.py          ← application complète (page unique, ~984 lignes)
src/config.py       ← constantes de chemins (PROJECT_ROOT, DATA_DIR, etc.)
.streamlit/
  config.toml       ← thème clair, couleur accent #1a1a1a (gris anthracite)
```

### Navigation

La navigation est assurée par une **barre horizontale** (`streamlit-option-menu`) qui remplace la sidebar Streamlit (masquée via CSS). Cinq pages sont accessibles :

| Icône | Page | Rôle |
|---|---|---|
| 🏠 | Le Projet | Tableau de bord introductif |
| 📊 | Données brutes | Visualisation EDA — assignment 4 |
| 🔧 | Feature Engineering | Transformations documentées — assignment 4 |
| 🤖 | Performances | Comparaison des modèles — assignment 4 |
| 🏷️ | Estimer un prix | Estimation interactive — assignment 5 |

### Fonctions utilitaires intégrées

| Fonction | Rôle |
|---|---|
| `geocode_adresse(adresse)` | Appel à l'API `api-adresse.data.gouv.fr` — géocodage de l'adresse |
| `prix_par_localisation(lat, lon)` | IDW (Inverse Distance Weighting) sur les 20 centroïdes d'arrondissements |
| `adj_etage_ascenseur(etage, sans_ascenseur)` | Bonus/malus d'étage et d'ascenseur selon les règles du marché parisien |
| `bonus_type_voie(street)` | Bonus/malus selon le type de voie (avenue +5%, impasse −5%, etc.) |
| `load_geo_trees()` | Chargement et cache des BallTrees géographiques (stations, parcs, commerces, parking) |
| `compute_geo_adjustments(lat, lon, trees)` | Calcul des ajustements de prix selon la proximité aux équipements |

---

## Description des pages

### Page 1 — Le Projet

Tableau de bord introductif qui présente le contexte, les chiffres-clés du projet et l'approche ML.

**Outputs affichés :**
- 4 métriques en haut de page : nombre de transactions analysées (156 077), prix médian Paris (10 300 €/m²), meilleur MAE (1 412 €/m²), meilleur R² (0,565)
- Tableau des 5 datasets sources (DVF, Stations IDF, Espaces verts, BPE 2024, Stationnement)
- Résumé de l'approche ML (type de problème, cible, split, features, métriques, meilleur modèle)

---

### Page 2 — Données brutes

Visualisation interactive du prix médian au m² par arrondissement, correspondant à la visualisation n°2 de l'assignment 4.

**Outputs affichés :**
- **Bar chart horizontal Plotly** trié par prix croissant, coloré sur une échelle Rouge→Vert (RdYlGn) — un survol de barre affiche le nom du quartier et le prix exact
- Ligne verticale pointillée indiquant la médiane parisienne (10 300 €/m²)
- 3 métriques : arrondissement le plus cher (6e — 14 646 €/m²), le moins cher (19e — 8 409 €/m²), écart max-min (6 237 €/m²)
- Encadré d'observation clé expliquant l'importance de l'arrondissement pour le modèle

---

### Page 3 — Feature Engineering

Documentation visuelle des transformations appliquées, correspondant à la visualisation n°3 de l'assignment 4. Organisée en deux onglets.

**Onglet "Log-transform de la cible" :**
- Deux histogrammes Plotly côte à côte : distribution brute de `prix_m2` (bleu) et distribution de `log(prix_m2)` (vert)
- Les histogrammes sont générés à partir d'une simulation lognormale paramétrée sur les statistiques DVF réelles (médiane 10 300 €/m², skewness ≈ 0,28)
- Légendes explicatives sous chaque graphique

**Onglet "Features construites" :**
- Tableau des 3 features créées pendant le feature engineering (`log_surface`, `surface_par_piece`, `arrondissement_prix_moyen`) avec leur formule et leur intérêt
- Tableau des 4 transformations testées et rejetées avec justification (StandardScaler, encodage ordinal, PCA, grille géographique)

---

### Page 4 — Performances

Comparaison des cinq modèles entraînés, correspondant à la visualisation n°4 de l'assignment 4.

**Outputs affichés :**
- **Tableau de données** listant les 5 modèles avec MAE, RMSE et R²
- **Bar chart MAE** (couleur Rouge→Vert inversé : rouge = mauvais, vert = bon) avec valeurs affichées au-dessus de chaque barre
- **Bar chart R²** (couleur Rouge→Vert : vert = bon) avec axe Y limité à 0,75
- Encadré d'interprétation : signification du MAE 1 412 €/m² (~14% du prix médian), explication du R² modéré (absence d'étage, état, balcon dans le DVF)

---

### Page 5 — Estimer un prix

Page principale de l'application. Permet d'obtenir une estimation du prix au m² selon deux modes, accessibles via deux onglets.

---

#### Onglet A — Par arrondissement (estimation rapide)

Formulaire en colonne gauche, résultat en colonne droite.

**Inputs utilisateur :**

| Input | Type | Valeur par défaut | Valeurs possibles |
|---|---|---|---|
| Arrondissement | Selectbox | 11e — Bastille/Oberkampf | 1e à 20e (avec nom de quartier) |
| Surface | Slider | 55 m² | 10 à 300 m², pas de 5 |
| Nombre de pièces | Slider | 3 | 1 à 10 |
| Étage | Selectbox | 2e | RDC à 9e |
| Sans ascenseur | Checkbox | Non | Oui / Non |
| Bouton "Estimer le prix" | Bouton primaire | — | — |

**Méthode de calcul :**

Le prix est estimé à partir du prix moyen de l'arrondissement (issu de l'EDA DVF), ajusté par plusieurs composantes :

```
prix_estimé = base_arrondissement
            + ajustement_surface     (−3,5 €/m² par m² au-dessus de 55)
            + ajustement_pièces      (fonction du ratio surface/pièces)
            + ajustement_annee       (+120 €/m² par an depuis 2022)
            + base × ajustement_étage_ascenseur
```

Le résultat est borné entre 3 000 et 25 000 €/m² (limites du dataset DVF).

**Outputs affichés :**
- **Métrique principale** : prix estimé au m², avec delta affichant la fourchette ±1 412 €/m² (MAE du meilleur modèle)
- **Métrique secondaire** : prix total pour la surface saisie
- **Légende** : détail du bonus/malus d'étage et d'ascenseur
- **Bar chart comparatif** : les 7 arrondissements proches numérotiquement, avec l'arrondissement sélectionné mis en vert, et une ligne horizontale indiquant le prix estimé
- Encadré de rappel de la marge d'erreur et du prix de base

---

#### Onglet B — Par adresse (estimation précise)

Mode d'estimation avancé qui tient compte de la micro-localisation à l'intérieur d'un arrondissement.

**Inputs utilisateur :**

| Input | Type | Valeur par défaut | Valeurs possibles |
|---|---|---|---|
| Adresse complète | Champ texte | — | Toute adresse parisienne |
| Surface | Slider | 55 m² | 10 à 300 m², pas de 5 |
| Nombre de pièces | Slider | 3 | 1 à 10 |
| Étage | Selectbox | 2e | RDC à 9e |
| Sans ascenseur | Checkbox | Non | Oui / Non |
| Bouton "Estimer par adresse" | Bouton primaire | — | — |

**Méthode de calcul :**

Le prix est calculé en cinq étapes successives :

1. **Géocodage** — l'adresse est envoyée à l'API `api-adresse.data.gouv.fr` (score de confiance minimum : 0,3). Un spinner indique l'attente.

2. **Prix de base IDW** — les coordonnées GPS sont pondérées par la distance inverse au carré aux 20 centroïdes d'arrondissements. Un appartement côté Parc Monceau dans le 17e reçoit ainsi une pondération plus forte vers le prix du 8e que vers le reste du 17e.

3. **Bonus type de voie** — le type de voie détecté dans l'adresse (avenue, boulevard, quai, impasse, passage…) génère un bonus ou malus sur le prix de base (avenue : +5%, impasse : −5%, quai : +4%, passage : −4%, etc.).

4. **Ajustements géographiques temps réel** — quatre BallTrees (chargés et mis en cache par `@st.cache_resource`) calculent les distances et densités d'équipements à la coordonnée du bien :

   | Équipement | Données source | Signal utilisé |
   |---|---|---|
   | 🚇 Métro/RER | IDFM open data | Distance à la station + nb de stations ≤ 500 m |
   | 🌳 Espaces verts | OpenData Paris | Distance au parc le plus proche (> 1 000 m²) |
   | 🛍️ Commerces | BPE 2024 INSEE | Nombre de commerces dans 500 m |
   | 🅿️ Stationnement | OpenData Paris | Nombre de places voiture dans 300 m |

5. **Ajustements bien** — mêmes ajustements que l'onglet A (surface, pièces, étage, ascenseur, tendance 2026).

**Outputs affichés :**
- Message de confirmation de l'adresse reconnue (ou message d'erreur si non trouvée)
- **Métrique principale** : prix estimé au m² avec fourchette ±1 412 €/m²
- **Métrique secondaire** : prix total pour la surface saisie
- **Tableau de décomposition** des ajustements :
  - Prix de base IDW
  - Bonus/malus type de voie (en %)
  - Ajustement Métro/RER (distance + nb stations)
  - Ajustement espaces verts (distance)
  - Ajustement commerces (nb dans 500 m)
  - Ajustement stationnement (nb dans 300 m)
  - Ajustement surface (en €/m²)
  - Ajustement pièces (en €/m²)
  - Ajustement étage et ascenseur (en %)
  - Tendance marché 2026 (en €/m²)
  - **Prix final estimé**
- **Carte interactive** (`st.map`) centrée sur l'adresse géocodée (zoom 14)
- Encadré explicatif de la méthode utilisée

---

## Justification des choix de conception

**Une seule page `app.py`** — pour un POC académique, regrouper le code dans un seul fichier simplifie la maintenance et la lecture. Aucun état persistant entre pages n'est nécessaire.

**Navigation horizontale** — la sidebar Streamlit est masquée et remplacée par `streamlit_option_menu` en orientation horizontale, ce qui donne plus d'espace vertical au contenu et une apparence plus proche d'une application web professionnelle.

**Deux modes d'estimation** — le mode "par arrondissement" répond à un besoin de rapidité (pas besoin de connaître l'adresse exacte), le mode "par adresse" illustre la valeur ajoutée du feature engineering géographique réalisé dans les notebooks.

**`@st.cache_resource` pour les BallTrees** — le chargement et la construction des quatre arbres géographiques est coûteux en temps (~2 s). Le cache garantit que cette opération n'est effectuée qu'une seule fois par session, quelle que soit la page consultée ou le nombre d'estimations demandées.

**Plotly au lieu de Matplotlib** — les graphiques Plotly sont interactifs (zoom, survol, téléchargement), ce qui est plus adapté à une application web qu'à un notebook.
