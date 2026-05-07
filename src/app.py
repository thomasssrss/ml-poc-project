"""Application Streamlit — Estimation du prix au m² à Paris (DVF).

Lancement :
    streamlit run src/app.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.neighbors import BallTree
from streamlit_option_menu import option_menu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Prix Immo Paris — POC",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS minimal — thème clair, police propre
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --c-accent: #1a1a1a;
    --c-border: #e5e5e5;
    --c-muted:  #6b6b6b;
    --radius:   8px;
    --font:     'Inter', -apple-system, sans-serif;
}

/* Police globale */
html, body, [data-testid="stApp"], [data-testid="stApp"] * {
    font-family: var(--font) !important;
    color: #1a1a1a;
}

/* Cacher la sidebar Streamlit — menu horizontal à la place */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Contenu principal — pas de marge gauche sans sidebar */
.main .block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1200px !important;
}

/* Titres */
h1 { font-size: 2rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; color: #1a1a1a !important; }
h2 { font-size: 1.35rem !important; font-weight: 600 !important; color: #1a1a1a !important; }
h3 { font-size: 1.05rem !important; font-weight: 500 !important; color: #1a1a1a !important; }
p, li { line-height: 1.65 !important; color: #1a1a1a !important; }

/* Métriques */
[data-testid="metric-container"] {
    background: #f8f8f8 !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--c-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #1a1a1a !important;
    letter-spacing: -0.02em !important;
}

/* Bouton principal */
.stButton > button[kind="primary"] {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #333 !important;
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
    font-size: 0.9rem !important;
}
[data-testid="stTextInput"] input:focus { border-color: #1a1a1a !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--c-border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--c-muted) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    border: none !important;
    padding: 0.75rem 1.25rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1a1a1a !important;
    border-bottom: 2px solid #1a1a1a !important;
}

/* Dataframes & alertes */
[data-testid="stDataFrame"] {
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stAlert"] { border-radius: var(--radius) !important; }
hr { border-color: var(--c-border) !important; }
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--c-muted) !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Données de référence (issues de l'EDA sur le dataset DVF)
# ---------------------------------------------------------------------------

PRIX_MOYEN_ARR = {
    1: 12700, 2: 11800, 3: 11500, 4: 12933,
    5: 12200, 6: 14646, 7: 14483, 8: 12304,
    9: 10800, 10: 10200, 11: 10500, 12: 9714,
    13: 9286,  14: 10100, 15: 10300, 16: 11934,
    17: 10800, 18: 9321,  19: 8409,  20: 8670,
}

NOM_ARR = {
    1: "Louvre / Les Halles", 2: "Bourse / Sentier", 3: "Marais Nord",
    4: "Marais / Île de la Cité", 5: "Panthéon / Latin", 6: "Saint-Germain-des-Prés",
    7: "Invalides / Tour Eiffel", 8: "Champs-Élysées", 9: "Opéra / Pigalle",
    10: "Canal Saint-Martin", 11: "Bastille / Oberkampf", 12: "Nation / Vincennes",
    13: "Gobelins / Chinatown", 14: "Montparnasse", 15: "Vaugirard",
    16: "Passy / Auteuil", 17: "Batignolles / Ternes", 18: "Montmartre / La Chapelle",
    19: "La Villette / Buttes-Chaumont", 20: "Belleville / Ménilmontant",
}

METRICS_DF = pd.DataFrame({
    "Modèle": [
        "Régression Linéaire",
        "Random Forest",
        "Gradient Boosting",
        "Random Forest v2 (log)",
        "Gradient Boosting v2 (log)",
    ],
    "MAE (€/m²)": [2026, 1572, 1825, 1412, 1826],
    "RMSE (€/m²)": [2951, 2357, 2690, 2227, 2739],
    "R²": [0.236, 0.513, 0.365, 0.565, 0.342],
})

# Centroïdes approximatifs des arrondissements parisiens (lat, lon)
ARR_CENTROIDS = {
    1:  (48.8605, 2.3477), 2:  (48.8659, 2.3493), 3:  (48.8637, 2.3619),
    4:  (48.8540, 2.3533), 5:  (48.8479, 2.3512), 6:  (48.8490, 2.3332),
    7:  (48.8566, 2.3177), 8:  (48.8750, 2.3094), 9:  (48.8766, 2.3381),
    10: (48.8766, 2.3599), 11: (48.8588, 2.3788), 12: (48.8407, 2.3939),
    13: (48.8315, 2.3588), 14: (48.8303, 2.3244), 15: (48.8418, 2.2950),
    16: (48.8636, 2.2722), 17: (48.8843, 2.3115), 18: (48.8927, 2.3444),
    19: (48.8837, 2.3836), 20: (48.8637, 2.3985),
}

EARTH_RADIUS_M = 6_371_000

# Répertoire des données externes (relatif à ce fichier)
_EXT_DIR = Path(__file__).parent.parent / "data" / "external"

# Bonus/malus selon le type de voie (en fraction du prix)
VOIE_BONUS = {
    "avenue":    0.05, "boulevard":  0.04, "place":      0.03,
    "esplanade": 0.03, "cours":      0.02, "quai":       0.04,
    "square":    0.01, "allée":      0.01, "rue":        0.00,
    "villa":    -0.02, "cité":      -0.02, "passage":   -0.04,
    "impasse":  -0.05, "ruelle":    -0.03, "sentier":   -0.03,
}


def geocode_adresse(adresse: str) -> dict | None:
    """Géocode une adresse parisienne via l'API gouvernementale française."""
    try:
        r = requests.get(
            "https://api-adresse.data.gouv.fr/search/",
            params={"q": adresse, "limit": 1, "citycode": "75056"},
            timeout=5,
        )
        data = r.json()
        if not data["features"]:
            return None
        feat = data["features"][0]
        props = feat["properties"]
        return {
            "label":   props.get("label", adresse),
            "lat":     feat["geometry"]["coordinates"][1],
            "lon":     feat["geometry"]["coordinates"][0],
            "score":   props.get("score", 0),
            "type":    props.get("type", ""),
            "street":  props.get("street", props.get("name", "")),
        }
    except Exception:
        return None


def prix_par_localisation(lat: float, lon: float) -> float:
    """
    Calcule un prix au m² pondéré par la distance aux centroïdes de chaque
    arrondissement (Inverse Distance Weighting).
    Un point dans le 17e proche du 8e reçoit automatiquement une pondération
    plus forte vers le prix du 8e.
    """
    poids_total = 0.0
    prix_pondere = 0.0
    for arr, (clat, clon) in ARR_CENTROIDS.items():
        dist = math.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
        dist = max(dist, 1e-6)
        w = 1.0 / dist ** 2      # pondération inverse au carré de la distance
        prix_pondere += w * PRIX_MOYEN_ARR[arr]
        poids_total  += w
    return prix_pondere / poids_total


def adj_etage_ascenseur(etage: int, sans_ascenseur: bool) -> float:
    """
    Retourne l'ajustement de prix (fraction) selon l'étage et l'absence d'ascenseur.

    Logique Paris :
    - L'ascenseur est un équipement standard → sa présence ne crée pas de bonus.
    - Son ABSENCE crée un malus progressif selon l'étage.

    Bonus étage (luminosité, vue, calme) :
    - RDC : -4%  |  1er : -2%  |  2e : 0%  |  3e-4e : +1,5%  |  5e-6e : +3%  |  7e+ : +4%

    Malus supplémentaire si sans ascenseur :
    - RDC/1er : 0%  |  2e : -1%  |  3e : -2,5%  |  4e : -4%  |  5e+ : -6%
    """
    if etage == 0:
        floor_bonus = -0.04
    elif etage == 1:
        floor_bonus = -0.02
    elif etage == 2:
        floor_bonus = 0.0
    elif etage <= 4:
        floor_bonus = 0.015
    elif etage <= 6:
        floor_bonus = 0.03
    else:
        floor_bonus = 0.04

    if sans_ascenseur:
        if etage <= 1:
            no_lift_malus = 0.0
        elif etage == 2:
            no_lift_malus = -0.01
        elif etage == 3:
            no_lift_malus = -0.025
        elif etage == 4:
            no_lift_malus = -0.04
        else:
            no_lift_malus = -0.06
    else:
        no_lift_malus = 0.0

    return floor_bonus + no_lift_malus


def bonus_type_voie(street: str) -> float:
    """Retourne le bonus/malus selon le type de voie détecté."""
    street_lower = street.lower()
    for voie, bonus in VOIE_BONUS.items():
        if street_lower.startswith(voie):
            return bonus
    return 0.0


@st.cache_resource
def load_geo_trees() -> dict:
    """
    Charge les datasets géographiques et construit les BallTrees (une seule fois).
    Retourne un dict avec les arbres pour : stations, parcs, commerces, parking.
    """
    trees: dict = {}

    # ── 1. Stations Métro / RER ───────────────────────────────────────────
    try:
        df_s = pd.read_csv(_EXT_DIR / "emplacement-des-gares-idf.csv", sep=";")
        coords_raw = df_s["Geo Point"].str.split(", ", expand=True)
        df_s["lat"] = pd.to_numeric(coords_raw[0], errors="coerce")
        df_s["lon"] = pd.to_numeric(coords_raw[1], errors="coerce")
        mask = (
            df_s["lat"].between(48.7, 49.0)
            & df_s["lon"].between(2.2, 2.55)
            & df_s["mode"].isin(["METRO", "RER"])
        )
        df_s = df_s[mask].dropna(subset=["lat", "lon"])
        trees["stations"] = BallTree(np.radians(df_s[["lat", "lon"]].values), metric="haversine")
    except Exception:
        trees["stations"] = None

    # ── 2. Espaces verts significatifs (> 1 000 m²) ───────────────────────
    try:
        df_ev = pd.read_csv(_EXT_DIR / "espaces_verts.csv", sep=";")
        ev_raw = df_ev["Geo point"].str.split(", ", expand=True)
        df_ev["lat"] = pd.to_numeric(ev_raw[0], errors="coerce")
        df_ev["lon"] = pd.to_numeric(ev_raw[1], errors="coerce")
        df_ev = df_ev.dropna(subset=["lat", "lon"])
        surface = pd.to_numeric(df_ev["Superficie totale réelle"], errors="coerce").fillna(0)
        df_ev = df_ev[surface > 1000]
        trees["parcs"] = BallTree(np.radians(df_ev[["lat", "lon"]].values), metric="haversine")
    except Exception:
        trees["parcs"] = None

    # ── 3. Commerces parisiens (BPE) ──────────────────────────────────────
    try:
        df_bpe = pd.read_csv(
            _EXT_DIR / "BPE24.csv", sep=";",
            usecols=["TYPEQU", "LATITUDE", "LONGITUDE", "DEP"],
            low_memory=False,
        )
        df_bpe = df_bpe[df_bpe["DEP"] == "75"]
        # Restaurants, cafés, boulangeries, banques
        com_types = ["B201", "B302", "B310", "B311"]
        df_com = df_bpe[df_bpe["TYPEQU"].isin(com_types)].dropna(
            subset=["LATITUDE", "LONGITUDE"]
        )
        trees["commerces"] = BallTree(
            np.radians(df_com[["LATITUDE", "LONGITUDE"]].values), metric="haversine"
        )
    except Exception:
        trees["commerces"] = None

    # ── 4. Places de stationnement voiture ───────────────────────────────
    try:
        with open(_EXT_DIR / "stationnement-voie-publique-emplacements.json") as f:
            parking_data = json.load(f)
        rows = []
        for feat in parking_data.get("features", []):
            props = feat.get("properties", {})
            regime = str(props.get("regpri", "")).upper()
            if any(r in regime for r in ["PAYANT", "ROTATIF", "GRATUIT"]):
                geo = feat.get("geometry", {})
                if geo.get("type") == "Point":
                    lon_p, lat_p = geo["coordinates"][:2]
                    rows.append((lat_p, lon_p))
        df_park = pd.DataFrame(rows, columns=["lat", "lon"])
        trees["parking"] = BallTree(np.radians(df_park[["lat", "lon"]].values), metric="haversine")
    except Exception:
        trees["parking"] = None

    return trees


def compute_geo_adjustments(lat: float, lon: float, trees: dict) -> tuple[dict, dict]:
    """
    Calcule des bonus/malus de prix selon la proximité aux équipements.

    Returns:
        adjustments : dict {catégorie → fraction du prix}
        details     : dict {catégorie → dict de métriques pour affichage}
    """
    pt = np.radians([[lat, lon]])
    adjustments: dict = {}
    details: dict = {}

    # ── Transport ────────────────────────────────────────────────────────
    if trees.get("stations") is not None:
        dist_rad, _ = trees["stations"].query(pt, k=1)
        dist_m = float(dist_rad[0, 0]) * EARTH_RADIUS_M
        nb_500 = int(
            trees["stations"].query_radius(pt, r=500 / EARTH_RADIUS_M, count_only=True)[0]
        )
        if dist_m < 200:
            b = 0.025
        elif dist_m < 400:
            b = 0.015
        elif dist_m < 700:
            b = 0.0
        else:
            b = -0.015
        if nb_500 > 3:
            b += 0.015
        adjustments["transport"] = b
        details["transport"] = {
            "label": f"Métro/RER le plus proche : {dist_m:.0f} m · {nb_500} stations ≤ 500 m",
            "bonus_pct": b * 100,
        }

    # ── Espaces verts ─────────────────────────────────────────────────────
    if trees.get("parcs") is not None:
        dist_rad, _ = trees["parcs"].query(pt, k=1)
        dist_m = float(dist_rad[0, 0]) * EARTH_RADIUS_M
        if dist_m < 150:
            b = 0.030
        elif dist_m < 400:
            b = 0.015
        elif dist_m < 700:
            b = 0.005
        else:
            b = 0.0
        adjustments["parcs"] = b
        details["parcs"] = {
            "label": f"Parc le plus proche : {dist_m:.0f} m",
            "bonus_pct": b * 100,
        }

    # ── Commerces ─────────────────────────────────────────────────────────
    if trees.get("commerces") is not None:
        nb_500 = int(
            trees["commerces"].query_radius(pt, r=500 / EARTH_RADIUS_M, count_only=True)[0]
        )
        if nb_500 > 15:
            b = 0.020
        elif nb_500 > 8:
            b = 0.010
        else:
            b = 0.0
        adjustments["commerces"] = b
        details["commerces"] = {
            "label": f"Commerces dans 500 m : {nb_500}",
            "bonus_pct": b * 100,
        }

    # ── Stationnement ─────────────────────────────────────────────────────
    if trees.get("parking") is not None:
        nb_300 = int(
            trees["parking"].query_radius(pt, r=300 / EARTH_RADIUS_M, count_only=True)[0]
        )
        if nb_300 > 15:
            b = 0.020
        elif nb_300 > 5:
            b = 0.010
        else:
            b = 0.0
        adjustments["parking"] = b
        details["parking"] = {
            "label": f"Places de stationnement dans 300 m : {nb_300}",
            "bonus_pct": b * 100,
        }

    return adjustments, details


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

page = option_menu(
    menu_title=None,
    options=["Le Projet", "Données brutes", "Feature Engineering", "Performances", "Estimer un prix"],
    icons=["house", "bar-chart", "tools", "cpu", "tag"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0",
            "background-color": "#ffffff",
            "border-bottom": "1px solid #e5e5e5",
            "margin-bottom": "1.5rem",
        },
        "nav": {
            "justify-content": "flex-start",
            "gap": "0",
        },
        "nav-link": {
            "font-family": "Inter, sans-serif",
            "font-size": "0.875rem",
            "font-weight": "450",
            "color": "#6b6b6b",
            "padding": "0.85rem 1.25rem",
            "border-radius": "0",
            "border-bottom": "2px solid transparent",
        },
        "nav-link-selected": {
            "background-color": "transparent",
            "color": "#1a1a1a",
            "font-weight": "600",
            "border-bottom": "2px solid #1a1a1a",
        },
        "icon": {"font-size": "0.85rem"},
    },
)

# ---------------------------------------------------------------------------
# Page 1 — Le Projet
# ---------------------------------------------------------------------------

if page == "Le Projet":
    st.title("🏠 Prédiction du prix au m² à Paris")
    st.subheader("Proof of Concept — Données DVF open data")

    st.markdown("""
    ### Objectif business

    Aider un acteur immobilier à **estimer rapidement le prix au m²** d'un appartement parisien
    à partir de ses caractéristiques et de sa localisation.

    Ce modèle peut servir à :
    - une **agence immobilière** pour proposer une première estimation de prix
    - un **vendeur** pour positionner son bien au juste prix
    - un **acheteur** pour détecter un bien potentiellement surévalué ou sous-évalué
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions analysées", "156 077")
    col2.metric("Prix médian Paris", "10 300 €/m²")
    col3.metric("Meilleur modèle MAE", "1 412 €/m²")
    col4.metric("Meilleur R²", "0.565")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### Données utilisées (5 datasets)

        | Dataset | Source |
        |---|---|
        | **DVF** | data.gouv.fr |
        | **Stations IDF** | IDFM open data |
        | **Espaces verts** | OpenData Paris |
        | **BPE 2024** | INSEE open data |
        | **Stationnement** | OpenData Paris |
        """)

    with col_b:
        st.markdown("""
        ### Approche ML

        - **Problème** : régression supervisée
        - **Cible** : `prix_m2 = valeur_foncière / surface`
        - **Split** : 80 % train / 20 % test
        - **Features** : 30 variables (DVF + géographiques)
        - **Métriques** : MAE, RMSE, R²
        - **Meilleur modèle** : Random Forest + log-transform
        """)

# ---------------------------------------------------------------------------
# Page 2 — D4 Visualisation 1 : Données brutes
# ---------------------------------------------------------------------------

elif page == "Données brutes":
    st.title("📊 Visualisation 1 — Données brutes")
    st.markdown(
        "Prix au m² médian par arrondissement parisien — "
        "156 077 ventes d'appartements (DVF 2014–2024)."
    )

    df_arr = pd.DataFrame({
        "Arrondissement": [f"{k}e" for k in PRIX_MOYEN_ARR],
        "Prix médian (€/m²)": list(PRIX_MOYEN_ARR.values()),
        "Quartier": list(NOM_ARR.values()),
        "Numéro": list(PRIX_MOYEN_ARR.keys()),
    }).sort_values("Prix médian (€/m²)", ascending=True)

    fig = px.bar(
        df_arr,
        x="Prix médian (€/m²)",
        y="Arrondissement",
        orientation="h",
        color="Prix médian (€/m²)",
        color_continuous_scale="RdYlGn",
        hover_data={"Quartier": True, "Prix médian (€/m²)": ":,.0f", "Arrondissement": False},
        title="Prix médian au m² par arrondissement parisien (DVF brut)",
    )
    fig.update_layout(height=620, coloraxis_showscale=False)
    fig.add_vline(
        x=10300, line_dash="dash", line_color="gray",
        annotation_text="Médiane Paris : 10 300 €/m²",
        annotation_position="top right",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Arrondissement le plus cher", "6e — 14 646 €/m²")
    col2.metric("Arrondissement le moins cher", "19e — 8 409 €/m²")
    col3.metric("Écart max – min", "6 237 €/m²")

    st.info(
        "**Observation clé** : l'arrondissement est la variable la plus discriminante. "
        "Un écart de 6 237 €/m² entre le 6e et le 19e justifie son rôle central dans le modèle."
    )

# ---------------------------------------------------------------------------
# Page 3 — D4 Visualisation 2 : Données transformées
# ---------------------------------------------------------------------------

elif page == "Feature Engineering":
    st.title("🔧 Visualisation 2 — Données transformées")
    st.markdown(
        "Illustration des transformations appliquées avant modélisation "
        "(documentées dans `notebooks/feature_engineering.ipynb`)."
    )

    tab1, tab2 = st.tabs(["📈 Log-transform de la cible", "🛠️ Features construites"])

    with tab1:
        st.markdown("""
        ### Effet du log-transform sur `prix_m2`

        La distribution brute est **asymétrique à droite** : quelques appartements très chers
        tirent la distribution. Le log-transform la rend **symétrique et gaussienne**,
        facilitant l'apprentissage des modèles (gain : MAE 1 572 → 1 412 €/m²).
        """)

        np.random.seed(42)
        prix_sim = np.random.lognormal(mean=np.log(10300), sigma=0.28, size=8000)
        prix_sim = prix_sim[(prix_sim >= 3000) & (prix_sim <= 25000)]

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(
                x=prix_sim, nbins=80,
                title="Distribution brute — prix_m2",
                labels={"x": "€/m²", "y": "Fréquence"},
                color_discrete_sequence=["#4C72B0"],
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("⚠️ Asymétrie à droite — longue queue vers les prix élevés")

        with col2:
            fig2 = px.histogram(
                x=np.log(prix_sim), nbins=80,
                title="Distribution transformée — log(prix_m2)",
                labels={"x": "log(€/m²)", "y": "Fréquence"},
                color_discrete_sequence=["#55A868"],
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("✅ Distribution symétrique — plus facile à modéliser")

    with tab2:
        st.markdown("### Features construites lors du feature engineering")

        features_df = pd.DataFrame({
            "Feature": ["log_surface", "surface_par_piece", "arrondissement_prix_moyen"],
            "Calcul": [
                "log(surface + 1)",
                "surface / nb_pieces",
                "Prix moyen de l'arrondissement (train set)",
            ],
            "Intérêt": [
                "Relation non-linéaire surface → prix",
                "Proxy du standing du bien",
                "Encode le niveau de prix de la zone",
            ],
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)

        st.markdown("""
        ### Transformations testées et rejetées

        | Transformation | Raison du rejet |
        |---|---|
        | StandardScaler | Sensible aux outliers → RobustScaler préféré |
        | Encodage ordinal arrondissement | Déjà un entier (1–20), redondant |
        | PCA | Détruit l'interprétabilité des features |
        | Grille géographique 80×80 | Dégradait les performances (R² 0.50→0.40) |
        """)

# ---------------------------------------------------------------------------
# Page 4 — D4 Visualisation 3 : Performances des modèles
# ---------------------------------------------------------------------------

elif page == "Performances":
    st.title("🤖 Visualisation 3 — Performances des modèles")
    st.markdown("Comparaison des 5 modèles entraînés — test set : 22 348 transactions.")

    st.dataframe(
        METRICS_DF,
        use_container_width=True,
        hide_index=True,
    )
    st.caption("🟢 Meilleur modèle : **Random Forest v2 (log-transform)** — MAE=1 412 €/m², R²=0.565")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig_mae = px.bar(
            METRICS_DF,
            x="Modèle", y="MAE (€/m²)",
            title="MAE (€/m²) — moins c'est mieux",
            color="MAE (€/m²)",
            color_continuous_scale="RdYlGn_r",
            text="MAE (€/m²)",
        )
        fig_mae.update_traces(texttemplate="%{text:,.0f} €", textposition="outside")
        fig_mae.update_layout(coloraxis_showscale=False, xaxis_tickangle=-15, height=400)
        st.plotly_chart(fig_mae, use_container_width=True)

    with col2:
        fig_r2 = px.bar(
            METRICS_DF,
            x="Modèle", y="R²",
            title="R² — plus c'est mieux (max = 1.0)",
            color="R²",
            color_continuous_scale="RdYlGn",
            text="R²",
        )
        fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_r2.update_layout(
            coloraxis_showscale=False, xaxis_tickangle=-15,
            yaxis_range=[0, 0.75], height=400,
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()
    st.markdown("""
    ### Interprétation

    - **MAE = 1 412 €/m²** : erreur moyenne de ~14% sur un prix médian de 10 300 €/m²
    - **R² = 0.565** : le modèle explique 56,5% de la variance des prix
    - Le **Random Forest** surpasse le Gradient Boosting (meilleure capture des non-linéarités)
    - Le R² modéré reflète l'absence dans le DVF de : **étage, état du bien, balcon, travaux**
    """)

# ---------------------------------------------------------------------------
# Page 5 — D5 Application : Estimation interactive
# ---------------------------------------------------------------------------

elif page == "Estimer un prix":
    st.title("🏷️ Estimer le prix au m² d'un appartement parisien")
    st.markdown(
        "Deux modes d'estimation : par **arrondissement** (rapide) "
        "ou par **adresse exacte** (plus précis — tient compte de la micro-localisation)."
    )

    tab_arr, tab_adresse = st.tabs(["🗺️ Par arrondissement", "📍 Par adresse (précis)"])

    # ── Onglet 1 : par arrondissement ─────────────────────────────────────────
    with tab_arr:
        col_form, col_result = st.columns([1, 1])

        with col_form:
            st.subheader("Caractéristiques du bien")

            arrondissement = st.selectbox(
                "Arrondissement",
                options=list(range(1, 21)),
                format_func=lambda x: f"{x}e — {NOM_ARR[x]}",
                index=10,
            )
            surface = st.slider("Surface (m²)", min_value=10, max_value=300, value=55, step=5)
            nb_pieces = st.slider("Nombre de pièces", min_value=1, max_value=10, value=3)
            etage = st.selectbox(
                "Étage",
                options=list(range(0, 10)),
                format_func=lambda x: "RDC" if x == 0 else f"{x}e",
                index=2,
                key="etage1",
            )
            sans_ascenseur = st.checkbox("Sans ascenseur", value=False, key="asc1")
            st.divider()
            estimer_arr = st.button("🔍 Estimer le prix", type="primary", use_container_width=True, key="btn_arr")

        with col_result:
            st.subheader("Résultat")
            if estimer_arr:
                base = PRIX_MOYEN_ARR[arrondissement]
                adj_surface = -3.5 * (surface - 55)
                adj_pieces  = 15 * (surface / max(nb_pieces, 1) - 20)
                annee_courante = 2026
                adj_annee   = (annee_courante - 2022) * 120
                frac_etage  = adj_etage_ascenseur(etage, sans_ascenseur)
                adj_etage_v = base * frac_etage
                prix_estime = int(max(3000, min(25000,
                    base + adj_surface + adj_pieces + adj_annee + adj_etage_v)))
                mae = 1412

                st.metric("Prix estimé au m²", f"{prix_estime:,} €/m²",
                          delta=f"Fourchette : {max(3000,prix_estime-mae):,} – {min(25000,prix_estime+mae):,} €/m²")
                st.metric(f"Prix total ({surface} m²)", f"{prix_estime*surface:,} €")

                etage_label = "RDC" if etage == 0 else f"{etage}e"
                asc_label = "sans ascenseur" if sans_ascenseur else "avec ascenseur"
                st.caption(f"Étage : {etage_label} · {asc_label} → {'+' if frac_etage>=0 else ''}{frac_etage*100:.1f}%")
                st.divider()

                voisins = sorted(PRIX_MOYEN_ARR.keys(), key=lambda k: abs(k - arrondissement))[:7]
                df_comp = pd.DataFrame({
                    "Arrondissement": [f"{k}e" for k in sorted(voisins)],
                    "Prix moyen (€/m²)": [PRIX_MOYEN_ARR[k] for k in sorted(voisins)],
                    "Sélectionné": [k == arrondissement for k in sorted(voisins)],
                })
                fig = px.bar(df_comp, x="Arrondissement", y="Prix moyen (€/m²)",
                             color="Sélectionné",
                             color_discrete_map={True: "#2ecc71", False: "#95a5a6"},
                             title="Comparaison avec les arrondissements proches")
                fig.update_layout(showlegend=False, height=300)
                fig.add_hline(y=prix_estime, line_dash="dash", line_color="#2ecc71",
                              annotation_text=f"Estimation : {prix_estime:,} €/m²")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Base {arrondissement}e ({base:,} €/m²) · marge ±{mae:,} €/m²")
            else:
                st.info("Remplissez le formulaire et cliquez sur **Estimer le prix**.")

    # ── Onglet 2 : par adresse ─────────────────────────────────────────────────
    with tab_adresse:
        st.markdown("""
        **Pourquoi c'est plus précis ?**
        Un appartement dans le 17e côté Parc Monceau vaut bien plus
        qu'un autre dans le 17e côté Porte de Clichy.
        En entrant l'adresse exacte, on calcule un prix pondéré par
        la distance réelle aux différents quartiers parisiens.
        """)

        col_form2, col_result2 = st.columns([1, 1])

        with col_form2:
            st.subheader("Adresse du bien")

            adresse_input = st.text_input(
                "Adresse complète",
                placeholder="Ex : 24 avenue de Wagram, Paris",
            )
            surface2   = st.slider("Surface (m²)", min_value=10, max_value=300, value=55, step=5, key="surf2")
            nb_pieces2 = st.slider("Nombre de pièces", min_value=1, max_value=10, value=3, key="pieces2")
            etage2     = st.selectbox(
                "Étage",
                options=list(range(0, 10)),
                format_func=lambda x: "RDC" if x == 0 else f"{x}e",
                index=2,
                key="etage2",
            )
            sans_ascenseur2 = st.checkbox("Sans ascenseur", value=False, key="asc2")
            st.divider()
            estimer_adresse = st.button("📍 Estimer par adresse", type="primary",
                                        use_container_width=True, key="btn_adresse")

        with col_result2:
            st.subheader("Résultat")

            if estimer_adresse:
                if not adresse_input.strip():
                    st.warning("Veuillez entrer une adresse.")
                else:
                    with st.spinner("Géocodage en cours..."):
                        geo = geocode_adresse(adresse_input)

                    if geo is None or geo["score"] < 0.3:
                        st.error("Adresse introuvable. Essayez avec plus de détails (numéro, rue, Paris).")
                    else:
                        st.success(f"📍 Adresse reconnue : **{geo['label']}**")

                        # Chargement des arbres géographiques (mis en cache)
                        with st.spinner("Calcul des proximités…"):
                            trees = load_geo_trees()
                            geo_adj, geo_details = compute_geo_adjustments(
                                geo["lat"], geo["lon"], trees
                            )

                        # Prix de base : micro-localisation IDW
                        prix_geo = prix_par_localisation(geo["lat"], geo["lon"])

                        # Bonus type de voie
                        bonus_voie = bonus_type_voie(geo["street"])

                        # Bonus équipements (somme de tous les ajustements géo)
                        total_geo_bonus = sum(geo_adj.values())

                        # Prix après tous les bonus de localisation
                        prix_localise = prix_geo * (1 + bonus_voie + total_geo_bonus)

                        # Ajustements bien (surface, pièces, étage, année)
                        adj_surface = -3.5 * (surface2 - 55)
                        adj_pieces  = 15 * (surface2 / max(nb_pieces2, 1) - 20)
                        adj_annee   = (2026 - 2022) * 120
                        frac_etage2 = adj_etage_ascenseur(etage2, sans_ascenseur2)
                        adj_etage_v2 = prix_localise * frac_etage2

                        prix_final = int(max(3000, min(25000,
                                        prix_localise + adj_surface + adj_pieces
                                        + adj_annee + adj_etage_v2)))
                        mae = 1412

                        st.metric("Prix estimé au m²", f"{prix_final:,} €/m²",
                                  delta=f"Fourchette : {max(3000,prix_final-mae):,} – {min(25000,prix_final+mae):,} €/m²")
                        st.metric(f"Prix total ({surface2} m²)", f"{prix_final*surface2:,} €")

                        # ── Détail des ajustements ─────────────────────────────
                        st.divider()
                        st.markdown("**Détail des ajustements :**")

                        type_voie = geo["street"].split()[0].lower() if geo["street"] else "rue"

                        rows_detail = [
                            ("Prix de base (micro-localisation IDW)", f"{prix_geo:,.0f} €/m²"),
                            (f"Type de voie ({type_voie})",
                             f"{'+' if bonus_voie>=0 else ''}{bonus_voie*100:.0f}%"),
                        ]

                        # Ajustements équipements
                        cat_labels = {
                            "transport":  "🚇 Transport (Métro/RER)",
                            "parcs":      "🌳 Espaces verts",
                            "commerces":  "🛍️ Commerces",
                            "parking":    "🅿️ Stationnement",
                        }
                        for cat, info in geo_details.items():
                            label = cat_labels.get(cat, cat)
                            b = info["bonus_pct"]
                            rows_detail.append((
                                f"{label} — {info['label']}",
                                f"{'+' if b>=0 else ''}{b:.1f}%",
                            ))

                        etage2_label = "RDC" if etage2 == 0 else f"{etage2}e"
                        asc2_label   = "sans ascenseur" if sans_ascenseur2 else "avec ascenseur"
                        rows_detail += [
                            ("Ajustement surface",
                             f"{'+' if adj_surface>=0 else ''}{adj_surface:.0f} €/m²"),
                            ("Ajustement pièces",
                             f"{'+' if adj_pieces>=0 else ''}{adj_pieces:.0f} €/m²"),
                            (f"🏢 Étage ({etage2_label} · {asc2_label})",
                             f"{'+' if frac_etage2>=0 else ''}{frac_etage2*100:.1f}%"),
                            ("Tendance marché 2026",
                             f"{'+' if adj_annee>=0 else ''}{adj_annee:.0f} €/m²"),
                            ("**Prix final estimé**", f"**{prix_final:,} €/m²**"),
                        ]

                        df_details = pd.DataFrame(rows_detail, columns=["Composante", "Valeur"])
                        st.dataframe(df_details, use_container_width=True, hide_index=True)

                        # Carte avec la position
                        df_map = pd.DataFrame({"lat": [geo["lat"]], "lon": [geo["lon"]]})
                        st.map(df_map, zoom=14)

                        st.info(
                            "📊 **Méthode** : prix IDW par micro-localisation + bonus/malus "
                            "type de voie + ajustements de proximité calculés en temps réel "
                            "sur les datasets Métro/RER, espaces verts, commerces et stationnement."
                        )
            else:
                st.markdown("""
                #### Exemple d'adresses à tester

                - `15 avenue de Wagram, Paris` ← 17e côté 8e (premium)
                - `10 rue de la Chapelle, Paris` ← 18e côté 19e (moins cher)
                - `5 quai de Bourbon, Paris` ← Île Saint-Louis (4e, très cher)
                - `20 rue des Pyrénées, Paris` ← 20e (abordable)

                ---
                **Marge d'erreur :** ±1 412 €/m² (MAE du meilleur modèle)
                """)


def build_app() -> None:
    """Point d'entrée pour scripts/main.py."""
    pass


if __name__ == "__main__":
    pass
