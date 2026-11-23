from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def model_path(name: str) -> Path:
    path = MODELS / name
    if not path.exists():
        pytest.skip(f"{name} not found; run scripts/export_models.py")
    return path
