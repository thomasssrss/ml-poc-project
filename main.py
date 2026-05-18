"""Point d'entrée racine — équivalent de `python scripts/main.py`.

Usage :
    python main.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    scripts_main = Path(__file__).parent / "scripts" / "main.py"
    result = subprocess.run([sys.executable, str(scripts_main)])
    sys.exit(result.returncode)
