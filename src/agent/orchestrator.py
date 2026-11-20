"""Bucle herramienta-LLM: el modelo decide, el código ejecuta."""

from __future__ import annotations

import json
from typing import Any

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import TOOL_SCHEMAS, ToolRegistry


class AgentOrchestrator:
    def __init__(self, llm_client: Any, tools: ToolRegistry, max_rounds: int = 3) -> None:
        self.llm = llm_client
        self.tools = tools
        self.max_rounds = max_rounds

    def run(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history or []:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": message})

        used_tools: list[dict[str, Any]] = []

        for _ in range(self.max_rounds):
            response = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            calls = response.get("tool_calls") or []

            if not calls:
                return {
                    "answer": response.get("content") or "",
                    "tool_calls": used_tools,
                    "messages": messages,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": response.get("content") or "",
                    "tool_calls": calls,
                }
            )

            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                raw_arguments = function.get("arguments", "{}")
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                    if not isinstance(arguments, dict):
                        raise ValueError("Los argumentos de la herramienta deben ser un objeto")
                except (ValueError, TypeError) as exc:
                    arguments = {}
                    result = {"status": "error", "error": f"Argumentos inválidos: {exc}"}
                else:
                    result = self.tools.dispatch(name, arguments)

                used_tools.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "status": result.get("status", "ok"),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        return {
            "answer": "No pude completar la consulta con las herramientas disponibles.",
            "tool_calls": used_tools,
            "messages": messages,
        }
