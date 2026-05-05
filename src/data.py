"""Student-owned dataset loading contract.

Charge le dataset DVF Paris, applique toutes les transformations de feature
engineering documentées dans notebooks/feature_engineering.ipynb, et retourne
le split (X_train, X_test, y_train, y_test) prêt pour la modélisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TARGET = "prix_m2"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Colonnes à supprimer : identifiants, fuites de données, constantes, quasi vides
COLS_TO_DROP = [
    # Identifiants
    "id_mutation", "numero_disposition", "id_parcelle",
    "ancien_id_parcelle", "numero_volume", "section_prefixe",
    # Fuite de données (prix_m2 est calculé depuis valeur_fonciere)
    "valeur_fonciere",
    # Adresse trop précise (cardinalité très élevée)
    "adresse_numero", "adresse_suffixe", "adresse_nom_voie", "adresse_code_voie",
    # Lots (quasi vides pour les appartements)
    "lot1_numero", "lot1_surface_carrez",
    "lot2_numero", "lot2_surface_carrez",
    "lot3_numero", "lot3_surface_carrez",
    "lot4_numero", "lot4_surface_carrez",
    "lot5_numero", "lot5_surface_carrez",
    "nombre_lots",
    # Constantes après filtrage (tous : Vente / Appartement)
    "nature_mutation", "type_local", "code_type_local",
    # Redondants avec code_postal (tous Paris 75)
    "code_commune", "nom_commune", "code_departement",
    "ancien_code_commune", "ancien_nom_commune",
    # Quasi vides pour les appartements
    "code_nature_culture", "nature_culture",
    "code_nature_culture_speciale", "nature_culture_speciale",
    "surface_terrain",
]

# Features géographiques produites par exploration_data.ipynb
GEO_FEATURES = [
    "distance_station_plus_proche", "nb_stations_500m", "nb_stations_1000m",
    "mode_station_plus_proche",
    "distance_ev_plus_proche", "surface_ev_plus_proche", "nb_ev_500m",
    "distance_education_plus_proche", "nb_education_500m",
    "distance_sante_plus_proche", "nb_sante_500m",
    "distance_commerce_plus_proche", "nb_commerce_500m",
    "distance_loisirs_plus_proche", "nb_loisirs_500m",
]


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _load_base(path: Path) -> pd.DataFrame:
    """Charge le dataset DVF nettoyé."""
    df = pd.read_csv(path, low_memory=False)
    return df


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes inutiles ou dangereuses (fuite de données)."""
    cols = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=cols)


def _extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait l'année et le mois depuis date_mutation."""
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df["annee"] = df["date_mutation"].dt.year
    df["mois"] = df["date_mutation"].dt.month
    df = df.drop(columns=["date_mutation"])
    return df


def _extract_arrondissement(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait le numéro d'arrondissement (1–20) depuis le code postal."""
    df = df.dropna(subset=["code_postal"]).copy()
    df["arrondissement"] = df["code_postal"].astype(float).astype(int).astype(str).str[-2:].astype(int)
    df = df.drop(columns=["code_postal"])
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes par la médiane."""
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(
        df["nombre_pieces_principales"].median()
    )
    # Supprime les lignes sans coordonnées GPS (très peu)
    df = df.dropna(subset=["latitude", "longitude"])
    return df


def _merge_geo_features(df: pd.DataFrame, enriched_path: Path) -> pd.DataFrame:
    """
    Fusionne les features géographiques si le dataset enrichi est disponible.
    Le dataset enrichi est produit par notebooks/exploration_data.ipynb.
    """
    if not enriched_path.exists():
        return df

    enriched = pd.read_csv(enriched_path, low_memory=False)
    cols_available = [c for c in GEO_FEATURES if c in enriched.columns]

    geo_df = enriched[["latitude", "longitude"] + cols_available].copy()
    df = df.merge(geo_df, on=["latitude", "longitude"], how="left")

    # Encodage OHE du mode de transport (nombre de modalités limité)
    if "mode_station_plus_proche" in df.columns:
        mode_dummies = pd.get_dummies(
            df["mode_station_plus_proche"], prefix="mode", drop_first=True
        )
        df = pd.concat([df, mode_dummies], axis=1)
        df = df.drop(columns=["mode_station_plus_proche"])

    return df


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conserve uniquement les features retenues + la cible.
    Supprime les colonnes restantes non utilisées (latitude/longitude
    gardées comme features de localisation brute).
    """
    base_features = [
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "arrondissement",
        "annee",
        "mois",
        "latitude",
        "longitude",
    ]
    geo_cols = [
        c for c in df.columns
        if c.startswith("distance_")
        or c.startswith("nb_")
        or c.startswith("surface_ev")
        or c.startswith("mode_")
    ]
    all_features = base_features + geo_cols
    all_features = [c for c in all_features if c in df.columns]

    df = df[all_features + [TARGET]].copy()
    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# Contrat public
# ---------------------------------------------------------------------------

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Charge, transforme et divise le dataset DVF Paris.

    Pipeline appliqué (documenté dans notebooks/feature_engineering.ipynb) :
    - Si dvf_paris_enriched.csv existe : chargement direct (toutes les features
      géographiques déjà calculées par exploration_data.ipynb)
    - Sinon : fallback sur dvf_paris_clean.csv (features DVF uniquement)
    Dans les deux cas : extraction temporelle, arrondissement, imputation,
    sélection des features finales, split 80/20.

    Returns:
        (X_train, X_test, y_train, y_test) — DataFrames et Series pandas.
    """
    enriched_path = DATA_DIR / "processed" / "dvf_paris_enriched.csv"
    base_path = DATA_DIR / "processed" / "dvf_paris_clean.csv"

    if enriched_path.exists():
        df = _load_base(enriched_path)
        # Le dataset enrichi a déjà prix_m2 et les features géo —
        # on supprime uniquement les colonnes inutiles encore présentes
        df = _drop_columns(df)
        df = _extract_temporal_features(df)
        df = _extract_arrondissement(df)
        df = _impute_missing(df)
        # Encodage OHE du mode de transport
        if "mode_station_plus_proche" in df.columns:
            mode_dummies = pd.get_dummies(
                df["mode_station_plus_proche"], prefix="mode", drop_first=True
            )
            df = pd.concat([df, mode_dummies], axis=1)
            df = df.drop(columns=["mode_station_plus_proche"])
    else:
        df = _load_base(base_path)
        df = _drop_columns(df)
        df = _extract_temporal_features(df)
        df = _extract_arrondissement(df)
        df = _impute_missing(df)

    df = _select_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test
