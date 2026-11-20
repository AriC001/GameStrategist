"""Carga única de embeddings, IDs, percentiles y ChromaDB de Module 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.paths import (
    CHROMA_DIR,
    EMBEDDING_CONFIG_PATH,
    EMBEDDING_IDS_PATH,
    EMBEDDINGS_PATH,
    MODULE_2_DIR,
)

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"
VARS_PERCENTILES = [
    "price_usd",
    "file_size_gb",
    "steam_positive_pct",
    "num_reviews",
    "owners_k",
    "peak_concurrent_players",
    "metacritic_score",
    "dlc_count",
]


def build_semantic_document(row: pd.Series) -> str:
    """Documento textual exactamente igual al usado en el notebook final."""
    genre_text = f"Es un videojuego del género {row['genre']}."
    indie_text = (
        "Es un videojuego desarrollado de forma independiente."
        if row["is_indie"] == 1
        else "No es un videojuego independiente."
    )
    multiplayer_text = (
        "Cuenta con modo multijugador."
        if row["has_multiplayer"] == 1
        else "No cuenta con modo multijugador."
    )
    early_access_text = (
        "Está disponible en acceso anticipado."
        if row["early_access"] == 1
        else "No está en acceso anticipado."
    )
    controller_text = (
        "Ofrece compatibilidad con controles."
        if row["controller_support"] == 1
        else "No ofrece compatibilidad con controles."
    )
    return " ".join(
        [
            f"{row['title']}.",
            genre_text,
            indie_text,
            multiplayer_text,
            early_access_text,
            controller_text,
        ]
    )


@dataclass
class Module2Artifacts:
    dataset: pd.DataFrame
    semantic_df: pd.DataFrame
    embeddings: np.ndarray
    game_ids: tuple[str, ...]
    id_a_pos: dict[str, int]
    percentiles: dict
    config: dict
    collection: Any
    embedder: Any | None

    @classmethod
    def load(
        cls,
        dataset: pd.DataFrame,
        base_dir: Path | None = None,
        load_embedder: bool = True,
    ) -> "Module2Artifacts":
        base = Path(base_dir) if base_dir else MODULE_2_DIR
        emb_path = base / EMBEDDINGS_PATH.name
        ids_path = base / EMBEDDING_IDS_PATH.name
        cfg_path = base / EMBEDDING_CONFIG_PATH.name
        chroma_path = base / "chroma_db"

        for path in (emb_path, ids_path, cfg_path, chroma_path):
            if not path.exists():
                raise FileNotFoundError(f"Falta artefacto de Module 2: {path}")

        semantic_df = dataset.copy()
        semantic_df["semantic_text"] = semantic_df.apply(build_semantic_document, axis=1)

        # Misma alineación explícita del notebook final.
        aligned = dataset.set_index("game_id").loc[semantic_df["game_id"]].reset_index()
        if not np.array_equal(
            aligned["game_id"].to_numpy(), semantic_df["game_id"].to_numpy()
        ):
            raise ValueError("Desalineación entre dataset y documentos semánticos")

        embeddings = np.load(emb_path)
        saved_ids = np.load(ids_path, allow_pickle=True).astype(str)
        if embeddings.shape != (len(aligned), 768):
            raise ValueError(
                f"Embeddings inesperados: {embeddings.shape}, se esperaba ({len(aligned)}, 768)"
            )
        if not np.array_equal(saved_ids, aligned["game_id"].to_numpy().astype(str)):
            raise ValueError("Los IDs guardados no coinciden con el orden de los embeddings")

        game_ids = tuple(saved_ids.tolist())
        id_a_pos = {gid: i for i, gid in enumerate(game_ids)}
        percentiles = {
            var: {
                f"P{q}": float(dataset[var].quantile(q / 100))
                for q in (10, 25, 50, 75, 90)
            }
            for var in VARS_PERCENTILES
        }
        config = json.loads(cfg_path.read_text(encoding="utf-8"))

        embedder = None
        if load_embedder:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedder = SentenceTransformer(
                config.get("modelo", MODELO_EMBEDDINGS), device=device
            )
            embedder.max_seq_length = int(config.get("max_seq_length", 256))

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(
            name="videojuegos_e5",
            metadata={"hnsw:space": "cosine"},
        )
        if collection.count() != len(aligned):
            raise ValueError(
                f"ChromaDB tiene {collection.count()} registros, se esperaban {len(aligned)}"
            )

        return cls(
            dataset=aligned,
            semantic_df=semantic_df,
            embeddings=embeddings,
            game_ids=game_ids,
            id_a_pos=id_a_pos,
            percentiles=percentiles,
            config=config,
            collection=collection,
            embedder=embedder,
        )
