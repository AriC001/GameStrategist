"""Carga única de los artefactos persistidos de Module 1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.paths import MODULE_1_CONFIG_PATH, MODULE_1_MODEL_PATH


@dataclass(frozen=True)
class Module1Artifacts:
    model: Any
    config: dict
    feature_cols: tuple[str, ...]
    threshold: float

    @classmethod
    def load(
        cls,
        model_path: Path | None = None,
        config_path: Path | None = None,
    ) -> "Module1Artifacts":
        model_file = Path(model_path) if model_path else MODULE_1_MODEL_PATH
        config_file = Path(config_path) if config_path else MODULE_1_CONFIG_PATH

        if not model_file.exists():
            raise FileNotFoundError(f"No existe el modelo: {model_file}")
        if not config_file.exists():
            raise FileNotFoundError(f"No existe la configuración: {config_file}")

        config = json.loads(config_file.read_text(encoding="utf-8"))
        features = tuple(config.get("features") or [])
        if len(features) != 11:
            raise ValueError(f"Se esperaban 11 features, hay {len(features)}")

        model = joblib.load(model_file)
        return cls(
            model=model,
            config=config,
            feature_cols=features,
            threshold=float(config["threshold"]),
        )
