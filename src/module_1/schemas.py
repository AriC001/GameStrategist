"""Esquemas de entrada del modelo predictivo."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

FEATURE_NAMES = (
    "genre",
    "release_year",
    "is_indie",
    "team_size",
    "price_usd",
    "has_multiplayer",
    "early_access",
    "controller_support",
    "languages_supported",
    "wishlists_k",
    "trailer_views_k",
)


class PredictRequest(BaseModel):
    """Acepta un game_id o las 11 features explícitas (o una mezcla)."""

    model_config = ConfigDict(extra="forbid")

    game_id: str | None = None
    genre: str | None = None
    release_year: int | None = None
    is_indie: int | None = Field(default=None, ge=0, le=1)
    team_size: int | None = Field(default=None, gt=0)
    price_usd: float | None = Field(default=None, ge=0)
    has_multiplayer: int | None = Field(default=None, ge=0, le=1)
    early_access: int | None = Field(default=None, ge=0, le=1)
    controller_support: int | None = Field(default=None, ge=0, le=1)
    languages_supported: int | None = Field(default=None, gt=0)
    wishlists_k: float | None = Field(default=None, ge=0)
    trailer_views_k: float | None = Field(default=None, ge=0)
