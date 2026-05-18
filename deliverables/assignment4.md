# Assignment 4 — Visualisations

## Rappel du projet

Prédiction du prix au m² des appartements parisiens à partir des données DVF enrichies avec des données géographiques externes (transports, espaces verts, équipements INSEE, stationnement).

---

## Notebooks et scripts — localisation et exécution

### Structure des notebooks

| Notebook | Rôle | Visualisations produites |
|---|---|---|
| `notebooks/exploration_data.ipynb` | Nettoyage DVF, EDA, enrichissement géographique | `plots/hist_prix_m2.png`, `plots/bar_prix_m2_arrondissement.png`, `plots/scatter_surface_valeur.png` |
| `notebooks/feature_engineering.ipynb` | Transformations, feature engineering documenté | `plots/fe_01_prix_par_annee.png`, `plots/scatter_surface_prix_m2.png`, `plots/fe_02_scaling_comparison.png`, `plots/fe_03_log_transform_target.png`, `plots/fe_04_pca_variance.png` |
| `notebooks/modelling.ipynb` | Entraînement des trois modèles, évaluation, comparaison | `plots/comparaison_modeles.png`, `plots/predictions_vs_reels.png`, `plots/feature_importances.png` |

### Comment exécuter les notebooks

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Placer les datasets sources dans les bons dossiers
#    data/raw/dvf.csv
#    data/external/emplacement-des-gares-idf.csv
#    data/external/espaces_verts.csv
#    data/external/BPE24.csv
#    data/external/stationnement-voie-publique-emplacements.json

# 3. Exécuter dans l'ordre
jupyter notebook notebooks/exploration_data.ipynb     # produit dvf_paris_enriched.csv
jupyter notebook notebooks/feature_engineering.ipynb  # produit dataset_ml.csv
jupyter notebook notebooks/modelling.ipynb            # entraîne les modèles et génère les plots
```

Toutes les visualisations sont sauvegardées automatiquement dans le dossier `plots/` lors de l'exécution des cellules concernées.

---

## Visualisation 1 — Données brutes : distribution des prix au m²

**Fichier :** `plots/hist_prix_m2.png`  
**Produit par :** `notebooks/exploration_data.ipynb` — section "EDA — Prix au m² à Paris"

### Objectif

Comprendre la forme et l'étendue de la variable cible (`prix_m2`) avant toute transformation. C'est la première étape indispensable de toute analyse : sans connaître la distribution de ce que l'on cherche à prédire, il est impossible de choisir des métriques ou des modèles adaptés.

### Choix du type de graphique

Un **histogramme** est le choix naturel pour visualiser la distribution d'une variable numérique continue. On y superpose deux lignes verticales : la médiane (en rouge) et la moyenne (en orange). L'écart entre ces deux marqueurs révèle immédiatement le degré d'asymétrie de la distribution.

### Interprétation

La distribution est **unimodale et légèrement asymétrique vers la droite** (skewness ≈ +0.5). Le gros du marché se situe entre 6 000 et 15 000 €/m², avec une médiane autour de 10 300 €/m² et une moyenne légèrement supérieure, signe d'une queue de distribution vers les prix élevés. Les bornes appliquées lors du nettoyage (3 000–25 000 €/m²) sont visibles : il n'y a pas de valeur extrême isolée. L'asymétrie reste modérée, ce qui justifie (voir feature engineering) de ne pas log-transformer la cible pour préserver l'interprétabilité des métriques en €/m².

---

## Visualisation 2 — Données brutes : prix médian par arrondissement

**Fichier :** `plots/bar_prix_m2_arrondissement.png`  
**Produit par :** `notebooks/exploration_data.ipynb` — section "EDA — Prix au m² à Paris"

### Objectif

Mettre en évidence les disparités spatiales du marché immobilier parisien entre arrondissements. C'est une visualisation directement liée à l'une des hypothèses centrales du projet : la localisation est l'un des principaux déterminants du prix.

### Choix du type de graphique

Un **bar chart horizontal** (barh) trié par prix croissant permet de classer visuellement les 20 arrondissements du moins cher au plus cher en un coup d'œil. Le format horizontal est préféré au format vertical car les labels d'arrondissements (codes postaux) sont lisibles sans rotation. Chaque barre porte la valeur numérique pour une lecture précise.

### Interprétation

L'écart entre le 1er et le 20e arrondissement dépasse **4 000 €/m²**. Les arrondissements du centre et de l'ouest (1er, 6e, 7e, 8e) sont systématiquement les plus chers, tandis que les arrondissements du nord-est (18e, 19e, 20e) sont les moins chers. Ce gradient géographique fort valide l'importance de la variable `arrondissement` comme feature prédictive. Cette observation a guidé la décision d'introduire un *target encoding* de l'arrondissement (prix moyen par arrondissement calculé uniquement sur le train set) lors de la phase de modélisation.

---

## Visualisation 3 — Feature engineering : évolution temporelle du prix m²

**Fichier :** `plots/fe_01_prix_par_annee.png`  
**Produit par :** `notebooks/feature_engineering.ipynb` — section "Création de nouvelles features"

### Objectif

Valider que l'extraction de la variable `annee` depuis la date de transaction apporte un signal prédictif réel. Si les prix médians sont stables dans le temps, la variable serait peu informative.

### Choix du type de graphique

Un **graphique en ligne avec marqueurs** (line plot) est le choix optimal pour une évolution temporelle ordinale. Contrairement à un bar chart qui traite chaque année de façon indépendante, le line plot met en évidence la tendance et les ruptures dans la série.

### Interprétation

Le graphique révèle une **hausse continue des prix entre 2014 et 2020** (+20 % environ), suivie d'une relative stabilisation et d'un début de correction à partir de 2022. Cette dynamique temporelle est directement liée au contexte macro-économique (faibles taux d'intérêt 2015–2021, puis hausse des taux). La variable `annee` est donc fortement corrélée au prix et justifie son inclusion comme feature. Ignorer la dimension temporelle reviendrait à mélanger dans le même modèle des transactions réalisées dans des conditions de marché très différentes.

---

## Visualisation 4 — Performances des modèles : comparaison MAE / RMSE / R²

**Fichier :** `plots/comparaison_modeles.png`  
**Produit par :** `notebooks/modelling.ipynb` — section "Comparaison des modèles"

### Objectif

Comparer les trois modèles (Régression Linéaire, Random Forest, Gradient Boosting) sur les trois métriques définies dans `assignment3.md` (MAE, RMSE, R²), en un seul graphique synthétique permettant une décision informée sur le choix du modèle final.

### Choix du type de graphique

Trois **bar charts** côte à côte (un par métrique), avec des couleurs identiques pour chaque modèle à travers les trois graphiques. Cette structure en triptyque permet de lire simultanément les trois dimensions de performance sans chercher dans un tableau. Les valeurs numériques sont affichées au-dessus de chaque barre pour éviter les erreurs de lecture visuelle.

### Interprétation

Les résultats sur le test set (22 348 lignes) sont les suivants :

| Modèle | MAE (€/m²) | RMSE (€/m²) | R² |
|---|---|---|---|
| Régression Linéaire (v1) | 2 026 | 2 951 | 0,236 |
| Random Forest (v1) | 1 572 | 2 357 | 0,513 |
| Gradient Boosting (v1) | 1 825 | 2 690 | 0,365 |
| **Random Forest v2 (log)** | **1 412** | **2 227** | **0,565** |
| Gradient Boosting v2 (log) | 1 826 | 2 739 | 0,342 |

Le **Random Forest v2 avec log-transformation de la cible** est le meilleur modèle : MAE de 1 412 €/m² et R² de 0,565. Cela signifie que le modèle se trompe en moyenne de 1 412 €/m², soit environ 13 % du prix médian (10 300 €/m²). La régression linéaire confirme son rôle de baseline : avec un MAE de 2 026 €/m², elle est 43 % moins précise que le meilleur modèle, ce qui justifie l'usage de méthodes non-linéaires.

---

## Visualisation 5 — Performances : prédictions vs prix réels et résidus

**Fichier :** `plots/predictions_vs_reels.png`  
**Produit par :** `notebooks/modelling.ipynb` — section "Prédictions vs Réels"

### Objectif

Aller au-delà des métriques agrégées (MAE, R²) pour diagnostiquer le comportement du modèle : est-ce qu'il se trompe de façon systématique pour certaines gammes de prix ? Est-ce que les erreurs sont symétriques ?

### Choix du type de graphique

Un **double panneau** associant un scatter plot (réels vs prédits) et un histogramme des résidus. Le scatter permet de voir la relation globale entre ce que le modèle prédit et ce qui est réel — un modèle parfait produirait tous les points sur la diagonale `y = x`. L'histogramme des résidus (erreur = réel − prédit) complète en montrant si les erreurs sont centrées sur zéro et symétriques. Pour le scatter, un sous-échantillon de 3 000 points est utilisé pour la lisibilité.

### Interprétation

Le scatter montre une corrélation positive nette entre prix prédits et prix réels, avec un nuage allongé autour de la diagonale. Deux déviations notables : le modèle **sous-estime les prix très élevés** (> 18 000 €/m²) et légèrement **sur-estime les prix très bas** (< 5 000 €/m²). Ce comportement est typique des Random Forests qui font une moyenne des sorties : ils ont du mal à extrapoler vers les extrêmes. L'histogramme des résidus est centré sur zéro et approximativement symétrique, sans biais systématique, ce qui est un bon signe. La queue légèrement droite correspond aux sous-estimations sur les biens haut de gamme.

---

## Visualisation 6 — Performances : importance des features (permutation importance)

**Fichier :** `plots/feature_importances.png`  
**Produit par :** `notebooks/modelling.ipynb` — section "Feature Importances"

### Objectif

Identifier quelles variables contribuent le plus à la performance du modèle. La permutation importance mesure la dégradation du R² quand on mélange aléatoirement les valeurs d'une feature — si le R² chute beaucoup, la feature était essentielle. C'est une étape indispensable pour valider les choix de feature engineering et comprendre le modèle.

### Choix du type de graphique

Un **bar chart horizontal trié par importance croissante** (de bas en haut : les features les plus importantes apparaissent en haut). Les 25 % de features les plus importantes sont colorées en vert foncé, les autres en bleu clair, pour mettre en évidence les leviers les plus critiques d'un coup d'œil.

### Interprétation

Les **5 features les plus importantes** sont systématiquement :
1. `arrondissement_prix_moyen` (target encoding de l'arrondissement) — de loin la variable la plus informative
2. `arrondissement` — même l'encodage ordinal seul est très utile
3. `surface_reelle_bati` — la surface reste un déterminant primaire du prix
4. `log_surface` — la log-surface apporte un signal complémentaire (relation non-linéaire)
5. `distance_station_plus_proche` — la proximité aux transports est le facteur géographique le plus discriminant

Ce résultat valide rétrospectivement trois décisions de feature engineering : l'enrichissement géographique par les transports, le target encoding de l'arrondissement (calculé sur le train set uniquement pour éviter la fuite), et la log-transformation de la surface. Les features de distance aux espaces verts et aux équipements (éducation, santé, commerces) contribuent de façon plus marginale mais mesurable.

---

## Justification de la pertinence des visualisations

Les six visualisations forment une narration cohérente qui suit le cycle de vie du projet :

1. **Visualisations 1 et 2 (données brutes)** → montrent l'état initial des données et valident l'hypothèse de départ : les prix parisiens sont hétérogènes, avec un fort gradient géographique. Sans ces graphiques, on ne peut pas justifier l'ajout des variables d'arrondissement et de localisation géographique.

2. **Visualisation 3 (feature engineering)** → démontre que la dimension temporelle est un signal fort, ce qui justifie l'extraction des variables `annee` et `mois`. Elle illustre directement comment une décision de feature engineering s'appuie sur une observation empirique plutôt que sur une supposition.

3. **Visualisations 4, 5 et 6 (performances)** → permettent une évaluation à plusieurs niveaux : la comparaison agrégée (graphique 4) donne une réponse claire sur quel modèle est le meilleur ; le diagnostic des résidus (graphique 5) identifie les zones d'erreur systématique ; les feature importances (graphique 6) ferment la boucle en reliant les performances aux choix de feature engineering.

Ensemble, ces visualisations permettent à un interlocuteur non-technique de comprendre le projet (graphiques 1, 2, 3) et à un interlocuteur technique d'évaluer la rigueur de la démarche (graphiques 4, 5, 6).
