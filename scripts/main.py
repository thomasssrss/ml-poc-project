from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent

config = _load_module("project_config", SCRIPT_DIR.parent / "src" / "config.py")
sys.modules["config"] = config
load_dotenv(config.ENV_FILE)
PROJECT_ROOT = config.PROJECT_ROOT
SRC_DIR = config.SRC_DIR
APP_ENTRYPOINT = config.APP_ENTRYPOINT
MODELS = config.MODELS
STREAMLIT_HOST = config.STREAMLIT_HOST
STREAMLIT_PORT = config.STREAMLIT_PORT

data_module = _load_module("project_data", SRC_DIR / "data.py")
metrics_module = _load_module("project_metrics", SRC_DIR / "metrics.py")
model_io_module = _load_module("project_model_io", SRC_DIR / "model_io.py")
results_module = _load_module("project_results", SRC_DIR / "results.py")

load_dataset_split = data_module.load_dataset_split
compute_metrics = metrics_module.compute_metrics
load_model = model_io_module.load_model
write_metrics = results_module.write_metrics


def _validate_models_config() -> None:
    if not MODELS:
        raise ValueError("config.MODELS is empty. Add your trained models first.")

    for model_key, model_config in MODELS.items():
        if "path" not in model_config:
            raise ValueError(
                f"Missing `path` for model `{model_key}` in config.MODELS."
            )


def _validate_app_entrypoint() -> None:
    app_module = _load_module("project_app", APP_ENTRYPOINT)
    if not hasattr(app_module, "build_app") or not callable(app_module.build_app):
        raise TypeError("app.build_app must be a callable Streamlit entry point.")


def _streamlit_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(SRC_DIR)]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _load_dataset() -> tuple[Any, Any, Any, Any]:
    dataset_split = load_dataset_split()
    if not isinstance(dataset_split, tuple) or len(dataset_split) != 4:
        raise ValueError(
            "data.load_dataset_split() must return exactly four values: "
            "(X_train, X_test, y_train, y_test)."
        )

    return dataset_split


def _evaluate_models(X_test: Any, y_test: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for model_key, model_config in MODELS.items():
        try:
            model = load_model(Path(model_config["path"]))
        except FileNotFoundError:
            print(f"  [skip] {model_key} — fichier modèle introuvable : {model_config['path']}")
            print(f"         Exécutez notebooks/modelling.ipynb pour régénérer les modèles.")
            continue

        if not hasattr(model, "predict"):
            print(f"  [skip] {model_key} — l'objet chargé n'expose pas de méthode predict.")
            continue

        try:
            y_pred = model.predict(X_test)
        except Exception as exc:
            print(f"  [skip] {model_key} — échec de la prédiction : {exc}")
            continue

        metrics = compute_metrics(y_test, y_pred)

        if not isinstance(metrics, dict) or not metrics:
            print(f"  [skip] {model_key} — compute_metrics() a retourné un résultat vide.")
            continue

        row: dict[str, object] = {
            "model_key": model_key,
            "model_name": model_config.get("name", model_key),
            "model_path": str(model_config["path"]),
        }
        for metric_name, metric_value in metrics.items():
            row[metric_name] = float(metric_value)

        rows.append(row)
        print(f"  [ok]   {model_config.get('name', model_key)} — {metrics}")

    return rows


def _launch_streamlit() -> None:
    if not APP_ENTRYPOINT.exists():
        raise FileNotFoundError(f"Streamlit entry point not found: {APP_ENTRYPOINT}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_ENTRYPOINT),
            "--server.address",
            STREAMLIT_HOST,
            "--server.port",
            str(STREAMLIT_PORT),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=_streamlit_env(),
    )


def main() -> None:
    print("=" * 60)
    print("  Prédiction du prix au m² — DVF Paris")
    print("=" * 60)

    _validate_app_entrypoint()

    # Évaluation des modèles — optionnelle si les données ne sont pas disponibles
    try:
        _validate_models_config()
        print("\n[1/3] Chargement du dataset...")
        _, X_test, _, y_test = _load_dataset()
        print(f"      {len(X_test):,} lignes de test chargées ({X_test.shape[1]} features)")

        print("\n[2/3] Évaluation des modèles...")
        metrics_rows = _evaluate_models(X_test, y_test)

        if metrics_rows:
            metrics_df = write_metrics(metrics_rows)
            print("\n      Résultats sauvegardés dans results/model_metrics.csv")
            print(metrics_df.to_string(index=False))
        else:
            print("      Aucun modèle n'a pu être évalué.")
            print("      Exécutez notebooks/modelling.ipynb pour entraîner les modèles.")

    except FileNotFoundError as exc:
        print(f"\n[!] Données manquantes — évaluation ignorée.")
        print(f"    {exc}")
        print("    → Consultez le README pour télécharger les données sources.")
        print("    → Exécutez les notebooks pour générer les datasets et modèles.")
    except Exception as exc:
        print(f"\n[!] Évaluation ignorée — {type(exc).__name__}: {exc}")

    print(f"\n[3/3] Lancement de l'application Streamlit...")
    print(f"      URL : http://{STREAMLIT_HOST}:{STREAMLIT_PORT}")
    print("=" * 60)

    _launch_streamlit()


if __name__ == "__main__":
    main()
