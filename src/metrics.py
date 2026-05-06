"""Student-owned metrics contract.

Calcule les métriques d'évaluation pour le problème de régression du prix au m²
des appartements parisiens (DVF).

Métriques retenues :
- MAE  : erreur absolue moyenne, interprétable directement en €/m²
- RMSE : racine de l'erreur quadratique moyenne, pénalise davantage les grandes erreurs
- R²   : part de variance expliquée par le modèle (0 = nul, 1 = parfait)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calcule MAE, RMSE et R² entre les valeurs réelles et prédites.

    Args:
        y_true: Valeurs réelles du prix au m² (array-like).
        y_pred: Valeurs prédites du prix au m² (array-like).

    Returns:
        Dictionnaire avec les clés "mae", "rmse", "r2".
        Toutes les valeurs sont des float.

    Exemple:
        >>> compute_metrics([10000, 12000], [10500, 11500])
        {'mae': 500.0, 'rmse': 500.0, 'r2': 0.75}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAE — erreur absolue moyenne (en €/m²)
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # RMSE — racine de l'erreur quadratique moyenne (en €/m²)
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

    # R² — coefficient de détermination
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2}
