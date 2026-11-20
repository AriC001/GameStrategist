"""Servidor del chat. Transporte interno, no una API REST pública."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import FRONTEND_DIR  # noqa: E402
from app.backend.container import ServiceContainer  # noqa: E402


class ChatHandler(BaseHTTPRequestHandler):
    container: ServiceContainer

    @classmethod
    def configure(cls, container: ServiceContainer) -> None:
        cls.container = container

    def _send(self, code: int, payload, content_type: str = "application/json; charset=utf-8") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            index = FRONTEND_DIR / "index.html"
            if not index.exists():
                self._send(404, {"error": "No existe app/frontend/index.html"})
                return
            self._send(200, index.read_bytes(), "text/html; charset=utf-8")
        elif path == "/health":
            self._send(200, self.container.status())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/chat":
            self._send(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            message = str(body.get("message", "")).strip()
            if not message:
                self._send(400, {"error": "El campo 'message' no puede estar vacío."})
                return

            result = self.container.orchestrator.run(message, body.get("history"))
            self._send(200, {
                "answer": result["answer"],
                "tool_calls": result["tool_calls"],
            })
        except Exception as exc:
            self._send(503, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[chat] {fmt % args}")


def _load_env_file() -> None:
    """Carga .env sin dependencias externas; el entorno real tiene prioridad."""
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    _load_env_file()
    load_embedder = os.getenv("LOAD_EMBEDDER", "1").lower() not in {"0", "false", "no"}
    container = ServiceContainer.load(load_embedder=load_embedder)
    ChatHandler.configure(container)

    port = int(os.getenv("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ChatHandler)
    print(f"GameStrategist listo en http://127.0.0.1:{port}")
    print(json.dumps(container.status(), ensure_ascii=False, indent=2))
    server.serve_forever()


if __name__ == "__main__":
    main()
