"""Turn PlantVillage folder names into readable labels and look up care information."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

INFO_FILE = Path(__file__).resolve().parents[2] / "data" / "disease_info.json"


def split_label(cls: str) -> tuple[str, str]:
    """'Corn_(maize)___Common_rust_' -> ('Corn (maize)', 'Common rust')"""
    plant, _, disease = cls.partition("___")
    clean = lambda s: re.sub(r"\s+", " ", s.replace("_", " ")).strip().rstrip(",")  # noqa: E731
    return clean(plant), clean(disease) or clean(plant)


def pretty_label(cls: str) -> str:
    plant, disease = split_label(cls)
    return f"{plant} – {disease}"


def is_healthy(cls: str) -> bool:
    return split_label(cls)[1].lower() == "healthy"


@lru_cache(maxsize=1)
def _info() -> dict:
    return json.loads(INFO_FILE.read_text(encoding="utf-8")) if INFO_FILE.exists() else {}


def disease_info(cls: str) -> dict:
    """Return {'type', 'about', 'actions'} for a class, with sensible fallbacks."""
    plant, disease = split_label(cls)
    if is_healthy(cls):
        return {"type": "Healthy", "about": f"No disease detected on this {plant.lower()} leaf.",
                "actions": ["Keep monitoring regularly", "Maintain good watering and spacing practices"]}
    return _info().get(cls, {
        "type": "Disease", "about": f"{disease} on {plant}.",
        "actions": ["Remove and destroy affected leaves",
                    "Consult your local agricultural extension officer for treatment"],
    })
