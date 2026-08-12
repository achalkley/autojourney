"""
Local HTTP bridge between the CLI and the AutoJourney Figma plugin.

The plugin's main sandbox (code.js) has no network access — that's a Figma
Plugin API restriction. Only its UI iframe (ui.html) can fetch(), gated by
manifest.json's networkAccess.devAllowedDomains. This server is what that
iframe talks to: it serves the session spec and screen images, and receives
progress/completion callbacks as the plugin builds the file.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Must match autojourney/figma_plugin/manifest.json's networkAccess.devAllowedDomains.
PLUGIN_BRIDGE_PORT = 8934
PLUGIN_COMPLETE_TIMEOUT_S = 600.0


class PluginBridgeServer:
    """Serves /spec and /image/<id> to the plugin's UI iframe, and receives
    /progress and /complete callbacks as it works."""

    def __init__(
        self,
        spec: dict[str, Any],
        images: dict[str, Path],
        progress_callback: Callable[[int, int, str], None] | None = None,
        port: int = PLUGIN_BRIDGE_PORT,
    ) -> None:
        self._spec = spec
        self._images = images
        self._progress_callback = progress_callback
        self._requested_port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._result: dict[str, Any] = {"done": False, "success": None, "error": None}

    @property
    def port(self) -> int:
        if self._server is not None:
            return self._server.server_address[1]
        return self._requested_port

    def start(self) -> None:
        spec = self._spec
        images = self._images
        progress_callback = self._progress_callback
        result = self._result

        class Handler(BaseHTTPRequestHandler):
            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

            def _send_json(self, status: int, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_empty(self, status: int) -> None:
                self.send_response(status)
                self._cors()
                self.end_headers()

            def do_OPTIONS(self) -> None:
                self._send_empty(204)

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/spec":
                    self._send_json(200, spec)
                    return
                if path.startswith("/image/"):
                    screen_id = path.removeprefix("/image/")
                    image_path = images.get(screen_id)
                    if image_path is None or not image_path.exists():
                        self._send_empty(404)
                        return
                    body = image_path.read_bytes()
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._send_empty(404)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}

                if path == "/progress":
                    if progress_callback:
                        progress_callback(
                            payload.get("done", 0), payload.get("total", 0), payload.get("detail", "")
                        )
                    self._send_empty(204)
                elif path == "/complete":
                    result["done"] = True
                    result["success"] = payload.get("success", False)
                    result["error"] = payload.get("error")
                    self._send_empty(204)
                else:
                    self._send_empty(404)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = HTTPServer(("localhost", self._requested_port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def wait_for_completion(self, timeout: float = PLUGIN_COMPLETE_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if self._result["done"]:
                    if not self._result["success"]:
                        raise RuntimeError(f"Figma plugin reported failure: {self._result['error']}")
                    return
                time.sleep(0.2)
            raise TimeoutError(
                "Timed out waiting for the AutoJourney Figma plugin to finish. "
                "Open the target file in Figma and run the plugin from Plugins → Development."
            )
        finally:
            self.stop()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=1)
