"""Esquemas y estructuras del buscador semántico."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

COLUMNAS_CANDIDATO = ["game_id", "title", "genre", "release_year", "price_usd"]


class TituloAmbiguo(Exception):
    def __init__(self, titulo: str, candidatos) -> None:
        self.titulo = titulo
        self.candidatos = candidatos
        super().__init__(
            f"'{titulo}' coincide con {len(candidatos)} videojuegos. "
            "Elegí un game_id o agregá género/año/precio_max."
        )


@dataclass
class Intencion:
    consulta: str
    generos: list = field(default_factory=list)
    genero_semantico: list = field(default_factory=list)
    atributos_semanticos: list = field(default_factory=list)
    obligatorios: dict = field(default_factory=dict)
    preferencias: dict = field(default_factory=dict)
    titulo: str | None = None
    advertencias: list = field(default_factory=list)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    game_id_referencia: str | None = None


class SimilarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id_referencia: str
    top_k: int = Field(default=10, ge=1, le=50)


class DetailsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str | None = None
    title: str | None = None
    genero: str | None = None
    anio: int | None = None
    precio_max: float | None = None


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_ids: list[str] = Field(min_length=2, max_length=5)
    fields: list[str] | None = None
    include_prediction: bool = False
