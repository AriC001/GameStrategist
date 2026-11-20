"""Cliente HTTP para el LLM local servido por Unsloth Studio.

No importa `unsloth`: habla contra cualquier endpoint compatible con
``/chat/completions`` de OpenAI. La URL y el modelo se configuran por entorno.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class LLMNotConfigured(RuntimeError):
    pass


class UnslothClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("UNSLOTH_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("UNSLOTH_MODEL", "")
        self.api_key = api_key or os.getenv("UNSLOTH_API_KEY", "")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> dict:
        if not self.base_url:
            raise LLMNotConfigured(
                "Falta UNSLOTH_BASE_URL. Configurá el entorno del LLM antes de usar /chat."
            )
        if not self.model:
            raise LLMNotConfigured("Falta UNSLOTH_MODEL.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        message = data["choices"][0]["message"]
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
            "raw": data,
        }

    def status(self) -> dict:
        return {
            "client": "UnslothClient",
            "base_url": self.base_url or None,
            "model": self.model or None,
            "configured": bool(self.base_url and self.model),
        }


class StaticClient:
    """Cliente falso, solo para probar el orquestador sin LLM local."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def chat(self, messages, tools=None, tool_choice=None) -> dict:
        return {"content": self.answer, "tool_calls": [], "raw": None}

    def status(self) -> dict:
        return {"client": "StaticClient", "configured": True}
