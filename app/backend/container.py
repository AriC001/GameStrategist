"""Contenedor de servicios: modelos y bases se cargan una sola vez."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.agent.llm_client import UnslothClient
from src.agent.orchestrator import AgentOrchestrator
from src.agent.tools import ToolRegistry
from src.module_1.loader import Module1Artifacts
from src.module_1.predictor import Predictor
from src.module_2.loader import Module2Artifacts
from src.module_2.search import SemanticSearch
from src.paths import DATASET_PATH


@dataclass
class ServiceContainer:
    dataset: pd.DataFrame
    predictor: Predictor
    search: SemanticSearch
    tools: ToolRegistry
    llm: Any
    orchestrator: AgentOrchestrator

    @classmethod
    def load(cls, load_embedder: bool = True, llm_client: Any | None = None) -> "ServiceContainer":
        dataset = pd.read_csv(DATASET_PATH)
        module_1 = Module1Artifacts.load()
        module_2 = Module2Artifacts.load(dataset, load_embedder=load_embedder)

        predictor = Predictor(module_1, dataset)
        search = SemanticSearch(module_2)
        tools = ToolRegistry(predictor, search)
        llm = llm_client if llm_client is not None else UnslothClient()
        orchestrator = AgentOrchestrator(llm, tools)

        return cls(
            dataset=dataset,
            predictor=predictor,
            search=search,
            tools=tools,
            llm=llm,
            orchestrator=orchestrator,
        )

    def status(self) -> dict:
        return {
            "status": "ok",
            "dataset": {
                "rows": int(len(self.dataset)),
                "columns": int(len(self.dataset.columns)),
            },
            "module_1": {
                "model": "RandomForestClassifier",
                "threshold": self.predictor.threshold,
                "features": len(self.predictor._artifacts.feature_cols),
            },
            "module_2": {
                "embeddings": list(self.search.embeddings.shape),
                "game_ids": len(self.search.id_a_pos),
                "chroma_records": self.search.a.collection.count(),
                "embedder_loaded": self.search.a.embedder is not None,
            },
            "llm": self.llm.status(),
        }
