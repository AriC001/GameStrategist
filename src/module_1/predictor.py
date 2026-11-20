"""Predicción de éxito usando únicamente el Random Forest persistido."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.module_1.loader import Module1Artifacts
from src.module_1.schemas import FEATURE_NAMES, PredictRequest


class Predictor:
    def __init__(self, artifacts: Module1Artifacts, dataset: pd.DataFrame) -> None:
        self._artifacts = artifacts
        self._dataset = dataset

    @property
    def threshold(self) -> float:
        return self._artifacts.threshold

    def predict(self, arguments: dict[str, Any] | PredictRequest) -> dict:
        request = (
            arguments
            if isinstance(arguments, PredictRequest)
            else PredictRequest.model_validate(arguments)
        )

        values: dict[str, Any] = {}
        if request.game_id is not None:
            rows = self._dataset.loc[self._dataset["game_id"] == request.game_id]
            if rows.empty:
                raise ValueError(f"game_id desconocido: {request.game_id}")
            row = rows.iloc[0]
            for name in FEATURE_NAMES:
                values[name] = row[name]

        for name in FEATURE_NAMES:
            supplied = getattr(request, name)
            if supplied is not None:
                values[name] = supplied

        missing = [name for name in FEATURE_NAMES if name not in values]
        if missing:
            raise ValueError(
                "Faltan features para predecir: "
                + ", ".join(missing)
                + ". Pasá game_id o completá todas las features."
            )

        features = pd.DataFrame(
            [[values[name] for name in FEATURE_NAMES]],
            columns=list(FEATURE_NAMES),
        )
        probability = float(self._artifacts.model.predict_proba(features)[0, 1])
        is_hit = probability >= self._artifacts.threshold

        return {
            "status": "ok",
            "game_id": request.game_id,
            "probability": round(probability, 6),
            "threshold": self._artifacts.threshold,
            "is_hit": bool(is_hit),
            "label": "hit" if is_hit else "no_hit",
            "model": "RandomForestClassifier (Module 1)",
            "features_used": list(FEATURE_NAMES),
            "note": (
                "La etiqueta se obtiene comparando la probabilidad con el threshold "
                "persistido; el LLM no interviene en el cálculo."
            ),
        }
