# Dataset — DVF Paris

## Source

Le projet utilisera les données DVF, pour Demandes de valeurs foncières.

## Objectif

L'objectif est de prédire le prix au m² des appartements à Paris.

## Filtrage prévu

Les données seront filtrées pour conserver uniquement :

- les ventes situées à Paris ;
- les biens de type appartement ;
- les lignes avec une surface réelle bâtie disponible ;
- les lignes avec une valeur foncière disponible ;
- les transactions cohérentes en termes de prix au m².

## Variable cible

La variable cible sera :

prix_m2 = valeur_fonciere / surface_reelle_batie