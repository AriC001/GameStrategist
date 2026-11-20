"""Predicción de éxito usando únicamente el Random Forest persistido."""

from __future__ import annotations

import numbers
from typing import Any

import pandas as pd

from src.module_1.loader import Module1Artifacts
from src.module_1.schemas import FEATURE_NAMES, PredictRequest
from src.utils import to_builtin

BINARY_FEATURES = (
    "is_indie",
    "has_multiplayer",
    "early_access",
    "controller_support",
)
INTEGER_FEATURES = ("release_year", "languages_supported")
NUMERIC_FEATURES = ("team_size", "price_usd", "wishlists_k", "trailer_views_k")

_EXPECTED = {
    "genre": "texto no vacío",
    "release_year": "entero (se aceptan años fuera del rango observado, ej. 2026)",
    "is_indie": "0, 1 o booleano",
    "team_size": "numérico > 0",
    "price_usd": "numérico >= 0",
    "has_multiplayer": "0, 1 o booleano",
    "early_access": "0, 1 o booleano",
    "controller_support": "0, 1 o booleano",
    "languages_supported": "entero > 0",
    "wishlists_k": "numérico >= 0",
    "trailer_views_k": "numérico >= 0",
}


class Predictor:
    def __init__(self, artifacts: Module1Artifacts, dataset: pd.DataFrame) -> None:
        self._artifacts = artifacts
        self._dataset = dataset

    @property
    def threshold(self) -> float:
        return self._artifacts.threshold

    def predict(self, arguments: dict[str, Any] | PredictRequest) -> dict:
        """Predice un juego del catálogo o uno definido por el usuario.

        Devuelve errores estructurados en vez de excepciones para que el agente
        sepa exactamente qué datos le faltan o cuáles son inválidos.
        """
        try:
            raw, fields = self._input_view(arguments)
        except TypeError as exc:
            return self._error("invalid_request", str(exc))

        has_id = "game_id" in fields and raw.get("game_id") is not None
        has_bundle = "game_features" in fields and raw.get("game_features") is not None
        flat = [name for name in FEATURE_NAMES if name in fields]
        modos = int(has_id) + int(has_bundle) + int(bool(flat))

        if modos != 1:
            return self._mode_error(modos, has_id, has_bundle, flat)

        if has_id:
            gid = str(raw["game_id"])
            rows = self._dataset.loc[self._dataset["game_id"] == gid]
            if rows.empty:
                return self._error(
                    "unknown_game_id",
                    f"game_id desconocido: {gid}",
                    mode="catalog",
                )
            fila = rows.iloc[0]
            values = {name: fila[name] for name in FEATURE_NAMES}
            ignored: list[str] = []
            mode = "catalog"
            game_id: str | None = gid
        elif has_bundle:
            bundle = raw["game_features"]
            if not isinstance(bundle, dict):
                return self._error(
                    "invalid_request",
                    "game_features debe ser un objeto con las características del juego.",
                    mode="custom_features",
                )
            values = {name: bundle[name] for name in FEATURE_NAMES if name in bundle}
            ignored = [name for name in bundle if name not in FEATURE_NAMES]
            mode = "custom_features"
            game_id = None
        else:  # legacy: 11 features sueltas, nunca mezcladas
            values = {name: raw[name] for name in flat}
            ignored = [
                name
                for name in raw
                if name not in FEATURE_NAMES and name not in {"game_id", "game_features"}
            ]
            mode = "custom_features"
            game_id = None

        cleaned, missing, invalid = self._validate_features(values)
        if missing or invalid:
            return self._incomplete_error(mode, missing, cleaned, ignored, invalid)

        features = pd.DataFrame(
            [[cleaned[name] for name in FEATURE_NAMES]],
            columns=list(FEATURE_NAMES),
        )
        probability = float(self._artifacts.model.predict_proba(features)[0, 1])
        return self._format_prediction(
            game_id,
            probability,
            cleaned,
            mode=mode,
            ignored_features=ignored,
        )

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

    @staticmethod
    def _input_view(arguments: dict[str, Any] | PredictRequest) -> tuple[dict, set]:
        if isinstance(arguments, PredictRequest):
            return arguments.model_dump(exclude_unset=True), set(arguments.model_fields_set)
        if isinstance(arguments, dict):
            return dict(arguments), set(arguments.keys())
        raise TypeError("Las argumentos de predicción deben ser un objeto o PredictRequest")

    @staticmethod
    def _as_number(value: Any) -> int | float:
        number = float(value)
        return int(number) if number.is_integer() else number

    @classmethod
    def _validate_features(cls, values: dict[str, Any]) -> tuple[dict, list, dict]:
        cleaned: dict[str, Any] = {}
        missing: list[str] = []
        invalid: dict[str, dict] = {}

        for name in FEATURE_NAMES:
            if name not in values or values[name] is None:
                missing.append(name)
                continue
            value = values[name]
            try:
                if name == "genre":
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError
                    cleaned[name] = value.strip()
                elif name in BINARY_FEATURES:
                    if isinstance(value, bool):
                        cleaned[name] = int(value)
                    elif isinstance(value, numbers.Integral) and int(value) in (0, 1):
                        cleaned[name] = int(value)
                    else:
                        raise ValueError
                elif name in INTEGER_FEATURES:
                    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                        raise ValueError
                    cleaned[name] = int(value)
                else:  # team_size, price_usd, wishlists_k, trailer_views_k
                    if isinstance(value, bool) or not isinstance(value, numbers.Real):
                        raise ValueError
                    number = float(value)
                    valido = number > 0 if name == "team_size" else number >= 0
                    if not valido:
                        raise ValueError
                    cleaned[name] = cls._as_number(value)
            except (ValueError, TypeError):
                invalid[name] = {"value": str(value), "expected": _EXPECTED[name]}

        return cleaned, missing, invalid

    @staticmethod
    def _error(code: str, message: str, mode: str | None = None) -> dict:
        return {
            "status": "error",
            "success": False,
            "error_code": code,
            "error": message,
            "mode": mode,
            "missing_features": [],
            "received_features": [],
            "invalid_features": {},
            "ignored_features": [],
        }

    @staticmethod
    def _mode_error(
        modos: int,
        has_id: bool,
        has_bundle: bool,
        flat: list[str],
    ) -> dict:
        if modos == 0:
            detail = "No se proporcionó ninguna modalidad."
            code = "missing_mode"
        else:
            partes = []
            if has_id:
                partes.append("game_id")
            if has_bundle:
                partes.append("game_features")
            if flat:
                partes.append("features legacy: " + ", ".join(flat))
            detail = "Se proporcionaron varias modalidades a la vez: " + "; ".join(partes)
            code = "ambiguous_mode"
        return {
            "status": "error",
            "success": False,
            "error_code": code,
            "error": detail,
            "mode": None,
            "missing_features": list(FEATURE_NAMES) if modos == 0 else [],
            "received_features": [],
            "invalid_features": {},
            "ignored_features": [],
            "message_to_agent": "Usá exactamente una modalidad: game_id o game_features.",
        }

    @staticmethod
    def _incomplete_error(
        mode: str,
        missing: list[str],
        cleaned: dict[str, Any],
        ignored: list[str],
        invalid: dict[str, dict],
    ) -> dict:
        return {
            "status": "error",
            "success": False,
            "error_code": "missing_features" if missing else "invalid_features",
            "error": "La predicción todavía no puede realizarse.",
            "mode": mode,
            "missing_features": list(missing),
            "received_features": list(cleaned),
            "invalid_features": invalid,
            "ignored_features": list(ignored),
            "message_to_agent": (
                "Pedí al usuario únicamente las características faltantes o inválidas. "
                "Conservá las recibidas y no imputes valores."
            ),
        }

    def _format_prediction(
        self,
        game_id: str | None,
        probability: float,
        features: dict[str, Any] | None = None,
        *,
        mode: str = "catalog",
        ignored_features: list[str] | None = None,
    ) -> dict:
        is_hit = probability >= self._artifacts.threshold
        label = "hit" if is_hit else "no_hit"
        probability_hit = round(probability, 6)
        usadas = to_builtin(features) if features is not None else {}
        return {
            # claves nuevas, contrato de la herramienta
            "status": "ok",
            "success": True,
            "mode": mode,
            "probability_hit": probability_hit,
            "prediction": label,
            "features_used": usadas,
            "ignored_features": list(ignored_features or []),
            # claves previas, conservadas para compare_games / search_hit_candidates
            "game_id": game_id,
            "probability": probability_hit,
            "threshold": self._artifacts.threshold,
            "is_hit": bool(is_hit),
            "label": label,
            "feature_names": list(FEATURE_NAMES),
            "features": usadas,
            "model": "RandomForestClassifier (Module 1)",
            "note": (
                "La etiqueta se obtiene comparando la probabilidad con el threshold "
                "persistido; el LLM no interviene en el cálculo."
            ),
        }
