"""Make `src/` importable when running scripts directly (python scripts/xyz.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
