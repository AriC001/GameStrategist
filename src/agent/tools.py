"""Herramientas expuestas al LLM y sus esquemas de function calling."""

from __future__ import annotations

from typing import Any, Callable

from src.module_1.predictor import Predictor
from src.module_2.schemas import (
    CompareRequest,
    DetailsRequest,
    SearchRequest,
    SimilarRequest,
)
from src.module_2.search import SemanticSearch
from src.utils import to_builtin

EXCLUDED_DETAIL_FIELDS = {"hit", "revenue_musd"}

_PREDICT_PROPERTIES = {
    "game_id": {"type": "string", "description": "ID del juego, por ejemplo G00001."},
    "genre": {"type": "string", "description": "Género principal del juego."},
    "release_year": {"type": "integer"},
    "is_indie": {"type": "integer", "enum": [0, 1]},
    "team_size": {"type": "integer"},
    "price_usd": {"type": "number"},
    "has_multiplayer": {"type": "integer", "enum": [0, 1]},
    "early_access": {"type": "integer", "enum": [0, 1]},
    "controller_support": {"type": "integer", "enum": [0, 1]},
    "languages_supported": {"type": "integer"},
    "wishlists_k": {"type": "number"},
    "trailer_views_k": {"type": "number"},
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "predict_game_success",
            "description": (
                "Estima la probabilidad de que un juego sea hit usando el Random Forest "
                "persistido de Module 1. Pasá game_id o las 11 features. La clasificación "
                "usa el threshold original del modelo; el LLM no la calcula."
            ),
            "parameters": {
                "type": "object",
                "properties": _PREDICT_PROPERTIES,
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_games",
            "description": (
                "Busca y rankea juegos en lenguaje natural con la estrategia híbrida de "
                "Module 2: filtros obligatorios, preferencias y similitud semántica."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta del usuario."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    "game_id_referencia": {
                        "type": "string",
                        "description": "Opcional; ID de referencia para similitud.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_games",
            "description": (
                "Devuelve juegos similares a un game_id usando la colección ChromaDB "
                "videojuegos_e5. Excluye el juego de referencia."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_id_referencia": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "required": ["game_id_referencia"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_game_details",
            "description": (
                "Devuelve la ficha de un juego por game_id o título. No expone hit ni "
                "revenue_musd. Si el título es ambiguo devuelve candidatas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_id": {"type": "string"},
                    "title": {"type": "string"},
                    "genero": {"type": "string"},
                    "anio": {"type": "integer"},
                    "precio_max": {"type": "number"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_games",
            "description": (
                "Compara entre 2 y 5 juegos por game_id. Puede incluir la predicción de "
                "Module 1 para cada uno. No usa hit ni revenue_musd."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 5,
                    },
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "include_prediction": {"type": "boolean", "default": False},
                },
                "required": ["game_ids"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolRegistry:
    def __init__(self, predictor: Predictor, search: SemanticSearch) -> None:
        self._predictor = predictor
        self._search = search
        self._handlers: dict[str, Callable[[dict], dict]] = {
            "predict_game_success": self.predict_game_success,
            "search_games": self.search_games,
            "find_similar_games": self.find_similar_games,
            "get_game_details": self.get_game_details,
            "compare_games": self.compare_games,
        }

    def dispatch(self, name: str, arguments: dict | None) -> dict:
        handler = self._handlers.get(name)
        if handler is None:
            return {"status": "error", "error": f"Herramienta desconocida: {name}"}
        try:
            return handler(arguments or {})
        except Exception as exc:  # el LLM recibe un error legible, no un traceback
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def predict_game_success(self, arguments: dict) -> dict:
        return self._predictor.predict(arguments)

    def search_games(self, arguments: dict) -> dict:
        request = SearchRequest.model_validate(arguments)
        return self._search.search(
            query=request.query,
            top_k=request.top_k,
            game_id_referencia=request.game_id_referencia,
        )

    def find_similar_games(self, arguments: dict) -> dict:
        request = SimilarRequest.model_validate(arguments)
        return self._search.similar(
            game_id_referencia=request.game_id_referencia,
            top_k=request.top_k,
        )

    def get_game_details(self, arguments: dict) -> dict:
        request = DetailsRequest.model_validate(arguments)
        resultado = self._search.get_details(
            game_id=request.game_id,
            title=request.title,
            genero=request.genero,
            anio=request.anio,
            precio_max=request.precio_max,
        )
        if resultado.get("game") is not None:
            resultado["game"] = self._visible(resultado["game"])
        return resultado

    def compare_games(self, arguments: dict) -> dict:
        request = CompareRequest.model_validate(arguments)
        juegos = []
        for gid in request.game_ids:
            resultado = self._search.get_details(game_id=gid)
            juego = self._visible(resultado["game"])
            if request.fields:
                juego = {k: juego[k] for k in request.fields if k in juego}
            item = {
                "game_id": gid,
                "title": resultado["game"].get("title"),
                "fields": to_builtin(juego),
            }
            if request.include_prediction:
                prediccion = self._predictor.predict({"game_id": gid})
                item["prediction"] = {
                    key: prediccion[key]
                    for key in ("probability", "threshold", "is_hit", "label")
                }
            juegos.append(item)

        diferencias = []
        campos = sorted({campo for juego in juegos for campo in juego["fields"]})
        for campo in campos:
            valores = {juego["game_id"]: juego["fields"].get(campo) for juego in juegos}
            if len({str(valor) for valor in valores.values()}) > 1:
                diferencias.append({"field": campo, "values": valores})

        return {
            "status": "ok",
            "games": juegos,
            "differences": diferencias,
            "excluded_fields": sorted(EXCLUDED_DETAIL_FIELDS),
        }

    @staticmethod
    def _visible(game: dict) -> dict:
        return {
            key: value
            for key, value in game.items()
            if key not in EXCLUDED_DETAIL_FIELDS
        }
