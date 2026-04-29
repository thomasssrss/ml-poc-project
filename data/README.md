# Dataset — DVF Paris

## Source

Le projet utilise un fichier DVF, pour Demandes de valeurs foncières.

Le fichier utilisé est stocké localement dans :

`data/raw/dvf.csv`

Il n'est pas versionné sur GitHub car il s'agit d'un fichier de données volumineux.

## Objectif du dataset

L'objectif est d'utiliser les transactions immobilières parisiennes pour prédire le prix au m² des appartements.

## Colonnes utiles

Les principales colonnes utilisées seront :

- `date_mutation` : date de la transaction ;
- `nature_mutation` : type de mutation, par exemple vente ;
- `valeur_fonciere` : prix de vente ;
- `code_postal` : code postal du bien ;
- `nom_commune` : commune ou arrondissement ;
- `code_departement` : département ;
- `type_local` : type de bien, par exemple appartement ;
- `surface_reelle_bati` : surface du bien ;
- `nombre_pieces_principales` : nombre de pièces ;
- `longitude` et `latitude` : coordonnées géographiques.

## Filtrage prévu

Les données seront filtrées pour conserver uniquement :

- les ventes ;
- les appartements ;
- les biens situés à Paris ;
- les lignes avec une valeur foncière positive ;
- les lignes avec une surface bâtie positive ;
- les observations avec un prix au m² cohérent.

## Variable cible

La variable cible sera le prix au m² :

`prix_m2 = valeur_fonciere / surface_reelle_bati`

## Localisation des fichiers

Le fichier brut doit être placé ici :

`data/raw/dvf.csv`

Le fichier nettoyé pourra être sauvegardé ici :

`data/processed/dvf_paris_clean.csv`

Le notebook d'exploration sera placé ici :

`notebooks/01_dvf_paris_eda.ipynb`