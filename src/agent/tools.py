"""Herramientas expuestas al LLM y sus esquemas de function calling."""

from __future__ import annotations

from typing import Any, Callable

from src.module_1.predictor import Predictor
from src.module_2.schemas import (
    CompareRequest,
    DetailsRequest,
    HitCandidatesRequest,
    SearchRequest,
    SimilarRequest,
)
from src.module_2.search import SemanticSearch
from src.utils import to_builtin

EXCLUDED_DETAIL_FIELDS = {"hit", "revenue_musd"}

_PREDICT_FEATURE_PROPERTIES = {
    "genre": {"type": "string", "description": "Género principal del juego (texto)."},
    "release_year": {
        "type": "integer",
        "description": "Año entero. Se aceptan valores fuera del rango del dataset, por ejemplo 2026.",
    },
    "is_indie": {"type": "integer", "enum": [0, 1], "description": "0/1 o booleano."},
    "team_size": {"type": "number", "description": "Numérico > 0."},
    "price_usd": {"type": "number", "description": "Numérico >= 0."},
    "has_multiplayer": {"type": "integer", "enum": [0, 1], "description": "0/1 o booleano."},
    "early_access": {"type": "integer", "enum": [0, 1], "description": "0/1 o booleano."},
    "controller_support": {
        "type": "integer",
        "enum": [0, 1],
        "description": "0/1 o booleano.",
    },
    "languages_supported": {"type": "integer", "description": "Entero > 0."},
    "wishlists_k": {"type": "number", "description": "Numérico >= 0."},
    "trailer_views_k": {"type": "number", "description": "Numérico >= 0."},
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "predict_game_success",
            "description": (
                "Estima la probabilidad de hit con el Random Forest persistido de Module 1. "
                "Usá EXACTAMENTE UNA modalidad: (A) game_id para juegos del catálogo, o "
                "(B) game_features con las 11 características para juegos nuevos/hipotéticos. "
                "No mezcles modalidades ni features sueltas. Si faltan datos, la herramienta "
                "devuelve missing_features: pedí solo esos datos y conservá los recibidos. "
                "Las features adicionales se ignoran y se informan en ignored_features. "
                "El threshold es un punto de corte, no un juicio de probabilidad alta."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Modalidad A: ID del juego, por ejemplo G00001.",
                    },
                    "game_features": {
                        "type": "object",
                        "description": (
                            "Modalidad B: características del juego nuevo o hipotético. "
                            "Las claves adicionales se ignoran y se devuelven en ignored_features."
                        ),
                        "properties": _PREDICT_FEATURE_PROPERTIES,
                        "additionalProperties": True,
                    },
                },
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
            "name": "search_hit_candidates",
            "description": (
                "Operación compuesta: ejecuta search_games con los criterios de la consulta "
                "y aplica el Random Forest de Module 1 a todos los candidatos en un solo "
                "lote. Devuelve la probabilidad de hit real de cada juego, el threshold y, "
                "por separado, el score de relevancia del buscador. La respuesta incluye "
                "`views` con el mismo pool ordenado por probabilidad y por precio: usalas "
                "cuando el usuario pida varias listas, sin repetir la búsqueda. Usala siempre que el "
                "usuario pida juegos con potencial de éxito, hits o probabilidad de éxito: "
                "reemplaza llamar a search_games y luego a predict_game_success varias veces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Criterios de búsqueda del usuario. No hace falta quitar "
                            "palabras como 'hit' o 'éxito'."
                        ),
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
                    "order_by": {
                        "type": "string",
                        "enum": [
                            "hit_probability",
                            "relevance",
                            "price_asc_then_probability",
                            "price_desc_then_probability",
                        ],
                        "default": "hit_probability",
                        "description": (
                            "Criterio de orden de la tabla. 'relevance' usa el ranking del "
                            "buscador; 'hit_probability' usa Module 1. Los valores "
                            "'price_asc_then_probability' y 'price_desc_then_probability' "
                            "ordenan primero por precio y desempatan por probabilidad. "
                            "Usalos cuando el usuario priorice precio; no reordienes vos."
                        ),
                    },
                    "min_probability": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Corte opcional de probabilidad; por defecto no filtra.",
                    },
                    "only_above_threshold": {
                        "type": "boolean",
                        "default": False,
                        "description": "Si es true, conserva solo juegos con is_hit=true.",
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
            "search_hit_candidates": self.search_hit_candidates,
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

    def search_hit_candidates(self, arguments: dict) -> dict:
        """Busca candidatos y predice hit en backend; el LLM solo redacta."""
        request = HitCandidatesRequest.model_validate(arguments)
        # Si manda precio, ampliamos el pool antes de ordenar: si no, un juego barato
        # con alta probabilidad puede quedar fuera por no estar en el top-k original.
        fetch_k = (
            min(50, request.top_k * 3)
            if request.order_by.startswith("price_")
            else request.top_k
        )
        busqueda = self.search_games({"query": request.query, "top_k": fetch_k})
        if busqueda.get("status") != "ok":
            # ambiguo o error: no hay una lista de candidatos sobre la que predecir
            return busqueda

        candidatos = busqueda.get("results") or []
        ids = [item["game_id"] for item in candidatos if item.get("game_id")]
        predicciones = self._predictor.predict_many(ids)["results"]
        por_id = {pred["game_id"]: pred for pred in predicciones}

        resultados = []
        for candidato in candidatos:
            gid = candidato.get("game_id")
            prediccion = por_id.get(gid)
            item = {
                "game_id": gid,
                "title": candidato.get("title"),
                "genre": candidato.get("genre"),
                "price_usd": candidato.get("price_usd"),
                "is_indie": candidato.get("is_indie"),
                "has_multiplayer": candidato.get("has_multiplayer"),
                "early_access": candidato.get("early_access"),
                "relevance_score": candidato.get("score"),
                "semantic_similarity": candidato.get("sim"),
                "por_que": candidato.get("por_que"),
            }
            if prediccion and prediccion.get("status") == "ok":
                item["hit_probability"] = prediccion["probability"]
                item["threshold"] = prediccion["threshold"]
                item["is_hit"] = prediccion["is_hit"]
                item["label"] = prediccion["label"]
                item["prediction_source"] = prediccion["model"]
                item["features_used"] = prediccion["features_used"]
                features = prediccion.get("features") or {}
                item["features"] = features
                # Las features reales son la fuente autorizada para describir el juego.
                item["has_multiplayer"] = features.get("has_multiplayer")
                item["early_access"] = features.get("early_access")
                item["is_indie"] = features.get("is_indie")
            else:
                item["hit_probability"] = None
                item["prediction_status"] = "error"
                item["prediction_error"] = (prediccion or {}).get(
                    "error", "predicción no disponible"
                )

            if request.min_probability is not None and (
                item["hit_probability"] is None
                or item["hit_probability"] < request.min_probability
            ):
                continue
            if request.only_above_threshold and item.get("is_hit") is not True:
                continue
            resultados.append(item)

        ordenados = self._ordenar_resultados(resultados, request.order_by)
        resultados = ordenados[: request.top_k]

        return {
            "status": "ok",
            "query": request.query,
            "strategy": busqueda.get("strategy"),
            "n_candidates": busqueda.get("n_candidates"),
            "order_by": request.order_by,
            "filters": {
                "min_probability": request.min_probability,
                "only_above_threshold": request.only_above_threshold,
            },
            "threshold": self._predictor.threshold,
            "top_k": request.top_k,
            "pool_size": len(ordenados),
            "results": resultados,
            "views": {
                "by_hit_probability": self._ordenar_resultados(
                    resultados, "hit_probability"
                ),
                "by_price_then_probability": self._ordenar_resultados(
                    resultados, "price_asc_then_probability"
                ),
            },
            "advertencias": busqueda.get("advertencias", []),
            "excluded_fields": sorted(EXCLUDED_DETAIL_FIELDS),
            "criterion": (
                "is_hit = hit_probability >= threshold. El threshold es un corte de "
                "clasificación validado en Module 1, no un umbral de 'probabilidad alta'."
            ),
            "note": (
                "hit_probability proviene del Random Forest (Module 1) con features reales "
                "del dataset. relevance_score y semantic_similarity provienen del buscador "
                "(Module 2) y no son probabilidades."
            ),
        }

    @staticmethod
    def _ordenar_resultados(resultados: list[dict], order_by: str) -> list[dict]:
        """Ordena una copia; en empates de precio manda hit_probability."""
        items = list(resultados)
        if order_by == "hit_probability":
            items.sort(
                key=lambda item: (
                    item["hit_probability"] is not None,
                    item["hit_probability"]
                    if item["hit_probability"] is not None
                    else -1.0,
                ),
                reverse=True,
            )
        elif order_by == "relevance":
            items.sort(key=lambda item: item["relevance_score"] or -1.0, reverse=True)
        else:  # price_asc_then_probability / price_desc_then_probability
            ascendente = order_by == "price_asc_then_probability"
            items.sort(
                key=lambda item: (
                    item["hit_probability"]
                    if item["hit_probability"] is not None
                    else -1.0
                ),
                reverse=True,
            )
            items.sort(
                key=lambda item: (
                    item["price_usd"] if item["price_usd"] is not None else float("inf")
                ),
                reverse=not ascendente,
            )
        return items

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
