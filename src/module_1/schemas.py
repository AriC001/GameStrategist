"""Esquemas de entrada del modelo predictivo."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

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
    """Una sola modalidad: `game_id`, `game_features` o las 11 features legacy.

    Los tipos se validan en `Predictor` para poder devolver errores estructurados
    (por ejemplo, un `release_year` futuro como 2026 no debe rechazarse por rango).
    """

    model_config = ConfigDict(extra="ignore")

    game_id: str | None = None
    game_features: dict[str, Any] | None = None

    # Compatibilidad legacy: se normalizan a game_features, nunca se mezclan.
    genre: Any = None
    release_year: Any = None
    is_indie: Any = None
    team_size: Any = None
    price_usd: Any = None
    has_multiplayer: Any = None
    early_access: Any = None
    controller_support: Any = None
    languages_supported: Any = None
    wishlists_k: Any = None
    trailer_views_k: Any = None
