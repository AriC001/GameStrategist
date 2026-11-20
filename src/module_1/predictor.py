"""Predicción de éxito usando únicamente el Random Forest persistido."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.module_1.loader import Module1Artifacts
from src.module_1.schemas import FEATURE_NAMES, PredictRequest
from src.utils import to_builtin


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
        return self._format_prediction(request.game_id, probability, values)

    def predict_many(self, game_ids: list[str]) -> dict:
        """Predice varios juegos de una sola llamada vectorizada.

        Las features siempre salen del dataset; el LLM no aporta valores.
        """
        ids = [str(gid) for gid in game_ids]
        if not ids:
            return {
                "status": "ok",
                "results": [],
                "model": "RandomForestClassifier (Module 1)",
                "threshold": self._artifacts.threshold,
                "features_used": list(FEATURE_NAMES),
            }

        indexado = self._dataset.set_index("game_id", drop=False)
        validos: list[tuple[str, pd.Series]] = []
        for gid in ids:
            if gid not in indexado.index:
                continue
            fila = indexado.loc[gid]
            if isinstance(fila, pd.DataFrame):  # game_id repetido en el dataset
                fila = fila.iloc[0]
            validos.append((gid, fila))

        por_fila = {gid: fila for gid, fila in validos}
        probabilidades: dict[str, float] = {}
        if validos:
            features = pd.DataFrame(
                [[fila[name] for name in FEATURE_NAMES] for _, fila in validos],
                columns=list(FEATURE_NAMES),
            )
            valores = self._artifacts.model.predict_proba(features)[:, 1]
            probabilidades = {
                gid: float(valor) for (gid, _), valor in zip(validos, valores)
            }

        resultados = []
        for gid in ids:
            if gid in probabilidades:
                fila = dict(por_fila[gid][list(FEATURE_NAMES)])
                resultados.append(self._format_prediction(gid, probabilidades[gid], fila))
            else:
                resultados.append(
                    {
                        "status": "error",
                        "game_id": gid,
                        "error": f"game_id desconocido: {gid}",
                    }
                )

        return {
            "status": "ok",
            "results": resultados,
            "model": "RandomForestClassifier (Module 1)",
            "threshold": self._artifacts.threshold,
            "features_used": list(FEATURE_NAMES),
            "note": (
                "Predicción batch del Random Forest con features reales del dataset; "
                "el LLM solo redacta el resultado."
            ),
        }

    def _format_prediction(
        self,
        game_id: str | None,
        probability: float,
        features: dict[str, Any] | None = None,
    ) -> dict:
        is_hit = probability >= self._artifacts.threshold
        return {
            "status": "ok",
            "game_id": game_id,
            "probability": round(probability, 6),
            "features": to_builtin(features) if features is not None else None,
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
