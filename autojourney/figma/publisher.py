"""
Figma integration via Figma's official remote MCP server.

Unlike the desktop/local MCP server, the remote server (https://mcp.figma.com/mcp)
does not accept a personal access token — it's OAuth-only. The first publish opens
a browser for a one-time authorization; the resulting token is cached on disk and
refreshed automatically after that.

The server exposes one general-purpose write tool, `use_figma`, which executes
JavaScript against the Figma Plugin API — there is no `create_frame` /
`create_connector` style REST surface. This module builds and sends that
JavaScript directly. Connector nodes are FigJam-only, so screen-to-screen edges
are drawn as simple arrow-capped vector paths instead.

Reference: https://developers.figma.com/docs/figma-mcp-server/
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx2
import networkx as nx
from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from autojourney import config
from autojourney.models import JourneySession

log = logging.getLogger(__name__)

OAUTH_CALLBACK_PORT = 8934
OAUTH_CALLBACK_TIMEOUT_S = 300

# Layout constants (Figma units ≈ pixels at 1x)
FRAME_W = 390        # iPhone 15 logical width
FRAME_H = 844        # iPhone 15 logical height
H_GAP = 120          # Horizontal gap between frames
V_GAP = 200          # Vertical gap between tree levels

# Keep each use_figma script small (figma-use guidance: work incrementally)
SCREENS_PER_BATCH = 3


# ──────────────────────────────────────────────────────────────────────────────
# OAuth: on-disk token storage + local loopback callback server
# ──────────────────────────────────────────────────────────────────────────────

class _FileTokenStorage(TokenStorage):
    """Persists OAuth tokens/client registration to disk so publishing doesn't
    re-open a browser window on every run."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("tokens"):
                self._tokens = OAuthToken.model_validate(data["tokens"])
            if data.get("client_info"):
                self._client_info = OAuthClientInformationFull.model_validate(data["client_info"])

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "tokens": self._tokens.model_dump(mode="json") if self._tokens else None,
            "client_info": self._client_info.model_dump(mode="json") if self._client_info else None,
        }))
        self._path.chmod(0o600)

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens
        self._save()

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info
        self._save()


class _OAuthCallbackServer:
    """Local loopback server that captures the single OAuth redirect, then stops."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._data: dict[str, str | None] = {"code": None, "state": None, "error": None}

    def start(self) -> None:
        data = self._data

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                query = parse_qs(urlparse(self.path).query)
                if "code" in query:
                    data["code"] = query["code"][0]
                    data["state"] = query.get("state", [None])[0]
                    body = b"<html><body>Figma authorized. You can close this tab.</body></html>"
                    self.send_response(200)
                elif "error" in query:
                    data["error"] = query["error"][0]
                    body = f"<html><body>Authorization failed: {query['error'][0]}</body></html>".encode()
                    self.send_response(400)
                else:
                    self.send_response(404)
                    body = b""
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = HTTPServer(("localhost", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def wait_for_code(self, timeout: float = OAUTH_CALLBACK_TIMEOUT_S) -> AuthorizationCodeResult:
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if self._data["code"]:
                    return AuthorizationCodeResult(code=self._data["code"], state=self._data["state"])
                if self._data["error"]:
                    raise RuntimeError(f"Figma authorization denied: {self._data['error']}")
                time.sleep(0.1)
            raise TimeoutError("Timed out waiting for Figma authorization in the browser")
        finally:
            self.stop()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=1)


def _build_oauth_provider(server_url: str) -> OAuthClientProvider:
    callback_server = _OAuthCallbackServer(OAUTH_CALLBACK_PORT)

    async def redirect_handler(authorization_url: str) -> None:
        log.info("Opening browser for Figma authorization...")
        webbrowser.open(authorization_url)

    async def callback_handler() -> AuthorizationCodeResult:
        callback_server.start()
        log.info("Waiting for Figma authorization in the browser...")
        return await asyncio.to_thread(callback_server.wait_for_code)

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=OAuthClientMetadata.model_validate({
            "client_name": "AutoJourney",
            "redirect_uris": [f"http://localhost:{OAUTH_CALLBACK_PORT}/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        }),
        storage=_FileTokenStorage(config.FIGMA_OAUTH_TOKEN_CACHE),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MCP tool call plumbing
# ──────────────────────────────────────────────────────────────────────────────

async def _call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    result = await session.call_tool(name, arguments)
    if result.is_error:
        text = "; ".join(c.text for c in result.content if c.type == "text")
        raise RuntimeError(f"Figma MCP tool '{name}' failed: {text or result.content}")
    if result.structured_content is not None:
        return result.structured_content
    texts = [c.text for c in result.content if c.type == "text"]
    if not texts:
        return None
    try:
        return json.loads(texts[0])
    except json.JSONDecodeError:
        return texts[0]


def _extract_upload_url(payload: Any) -> str:
    """upload_assets' exact response shape isn't part of its published schema,
    so search for the first http(s) URL rather than assuming one key path."""

    def _walk(node: Any) -> str | None:
        if isinstance(node, str) and node.startswith("http"):
            return node
        if isinstance(node, dict):
            for v in node.values():
                found = _walk(v)
                if found:
                    return found
        if isinstance(node, list):
            for v in node:
                found = _walk(v)
                if found:
                    return found
        return None

    url = _walk(payload)
    if not url:
        raise RuntimeError(f"Could not find an upload URL in upload_assets response: {payload!r}")
    return url


# ──────────────────────────────────────────────────────────────────────────────
# Plugin API script builders
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_page_script(page_name: str) -> str:
    return f"""
const pageName = {json.dumps(page_name)};
let page = figma.root.children.find(p => p.name === pageName);
if (!page) {{
  page = figma.createPage();
  page.name = pageName;
}}
await figma.setCurrentPageAsync(page);
return {{ pageId: page.id }};
"""


def _create_screens_script(page_id: str, specs: list[dict[str, Any]]) -> str:
    return f"""
const page = await figma.getNodeByIdAsync({json.dumps(page_id)});
await figma.setCurrentPageAsync(page);
const specs = {json.dumps(specs)};
const created = [];
for (const s of specs) {{
  const frame = figma.createFrame();
  frame.name = s.name;
  frame.resize(s.w, s.h);
  frame.x = s.x;
  frame.y = s.y;
  frame.fills = [{{ type: 'SOLID', color: {{ r: 0.94, g: 0.94, b: 0.94 }} }}];
  page.appendChild(frame);

  const label = figma.createText();
  await figma.loadFontAsync(label.fontName);
  label.characters = s.label;
  label.fontSize = 13;
  label.fontName = {{ ...label.fontName, style: 'Bold' }};
  label.resize(s.w, label.height);
  label.x = s.x;
  label.y = s.y + s.h + 8;
  page.appendChild(label);

  const entry = {{ screenId: s.id, frameId: frame.id, labelId: label.id }};

  if (s.action) {{
    const actionText = figma.createText();
    await figma.loadFontAsync(actionText.fontName);
    actionText.characters = '\\u21B3 ' + s.action;
    actionText.fontSize = 11;
    actionText.resize(s.w, actionText.height);
    actionText.x = s.x;
    actionText.y = s.y + s.h + 28;
    page.appendChild(actionText);
    entry.actionLabelId = actionText.id;
  }}

  created.push(entry);
}}
return {{ created }};
"""


def _connectors_script(page_id: str, edges: list[dict[str, Any]]) -> str:
    return f"""
const page = await figma.getNodeByIdAsync({json.dumps(page_id)});
await figma.setCurrentPageAsync(page);
const edges = {json.dumps(edges)};
const createdIds = [];
for (const e of edges) {{
  const fromNode = await figma.getNodeByIdAsync(e.fromId);
  const toNode = await figma.getNodeByIdAsync(e.toId);
  if (!fromNode || !toNode) continue;

  const startX = fromNode.x + fromNode.width / 2;
  const startY = fromNode.y + fromNode.height;
  const endX = toNode.x + toNode.width / 2;
  const endY = toNode.y;

  const minX = Math.min(startX, endX);
  const minY = Math.min(startY, endY);
  const w = Math.max(Math.abs(endX - startX), 1);
  const h = Math.max(Math.abs(endY - startY), 1);
  const localStartX = startX - minX;
  const localStartY = startY - minY;
  const localEndX = endX - minX;
  const localEndY = endY - minY;

  const vector = figma.createVector();
  vector.resize(w, h);
  vector.x = minX;
  vector.y = minY;
  vector.vectorPaths = [{{
    windingRule: 'NONE',
    data: `M ${{localStartX}} ${{localStartY}} L ${{localEndX}} ${{localEndY}}`,
  }}];
  vector.strokeWeight = 2;
  vector.strokes = [{{ type: 'SOLID', color: {{ r: 0.35, g: 0.4, b: 0.9 }} }}];
  vector.strokeCap = 'ARROW_LINES';
  page.appendChild(vector);
  createdIds.push(vector.id);

  if (e.label) {{
    const text = figma.createText();
    await figma.loadFontAsync(text.fontName);
    text.characters = e.label;
    text.fontSize = 10;
    text.x = (startX + endX) / 2;
    text.y = (startY + endY) / 2;
    page.appendChild(text);
    createdIds.push(text.id);
  }}
}}
return {{ createdIds }};
"""


# ──────────────────────────────────────────────────────────────────────────────
# Main publish function
# ──────────────────────────────────────────────────────────────────────────────

async def _publish_async(
    session: JourneySession,
    graph: nx.DiGraph,
    layout: dict[str, tuple[float, float]],
    page_name: str | None,
    progress_callback,
) -> str:
    file_key = config.FIGMA_FILE_KEY
    target_page = page_name or config.FIGMA_PAGE_NAME
    server_url = config.FIGMA_MCP_SERVER_URL

    oauth_auth = _build_oauth_provider(server_url)

    log.info("Connecting to Figma MCP server: %s", server_url)
    async with httpx2.AsyncClient(auth=oauth_auth, follow_redirects=True) as http_client:
        async with streamable_http_client(url=server_url, http_client=http_client) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as mcp_session:
                await mcp_session.initialize()

                log.info("Publishing journey map to Figma (file: %s, page: %s)", file_key, target_page)
                page_result = await _call_tool(mcp_session, "use_figma", {
                    "fileKey": file_key,
                    "code": _ensure_page_script(target_page),
                    "description": f"Ensure Figma page '{target_page}' exists for the journey map",
                    "skillNames": "figma-use",
                })
                page_id = page_result["pageId"]

                if layout:
                    min_x = min(x for x, _ in layout.values())
                    min_y = min(y for _, y in layout.values())
                    layout = {n: (x - min_x + 100, y - min_y + 100) for n, (x, y) in layout.items()}

                screen_map = {s.screen_id: s for s in session.screens}
                node_map: dict[str, str] = {}
                total = len(session.screens)
                done = 0

                for batch_start in range(0, len(session.screens), SCREENS_PER_BATCH):
                    batch = session.screens[batch_start:batch_start + SCREENS_PER_BATCH]
                    specs = []
                    for i, screen in enumerate(batch, start=batch_start):
                        lx, ly = layout.get(screen.screen_id, (i * (FRAME_W + H_GAP) + 100, 100))
                        frame_h = FRAME_H
                        try:
                            import cv2
                            img = cv2.imread(str(screen.image_path))
                            if img is not None:
                                native_h, native_w = img.shape[:2]
                                frame_h = int(FRAME_W * native_h / native_w)
                        except Exception:
                            pass
                        frame_label = screen.screen_name or f"Screen {screen.screen_id}"
                        specs.append({
                            "id": screen.screen_id,
                            "name": frame_label,
                            "x": lx,
                            "y": ly,
                            "w": FRAME_W,
                            "h": frame_h,
                            "label": f"{screen.app_name} — {screen.screen_name}" if screen.app_name else frame_label,
                            "action": screen.inferred_action or None,
                        })

                    batch_result = await _call_tool(mcp_session, "use_figma", {
                        "fileKey": file_key,
                        "code": _create_screens_script(page_id, specs),
                        "description": f"Create {len(specs)} screen frame(s) for the journey map",
                        "skillNames": "figma-use",
                    })
                    for entry in batch_result["created"]:
                        node_map[entry["screenId"]] = entry["frameId"]

                    for entry in batch_result["created"]:
                        screen = screen_map[entry["screenId"]]
                        upload_result = await _call_tool(mcp_session, "upload_assets", {
                            "fileKey": file_key,
                            "nodeId": entry["frameId"],
                            "count": 1,
                        })
                        upload_url = _extract_upload_url(upload_result)
                        image_bytes = Path(screen.image_path).read_bytes()
                        upload_resp = await http_client.post(
                            upload_url, content=image_bytes, headers={"Content-Type": "image/png"}
                        )
                        upload_resp.raise_for_status()

                        done += 1
                        if progress_callback:
                            progress_callback(done, total, f"Placed: {screen.screen_name or screen.screen_id}")

                # Draw connectors between screens now that every frame exists
                edges = [
                    {"fromId": node_map[e.from_screen_id], "toId": node_map[e.to_screen_id], "label": e.interaction_label}
                    for e in session.edges
                    if e.from_screen_id in node_map and e.to_screen_id in node_map
                ]
                if edges:
                    await _call_tool(mcp_session, "use_figma", {
                        "fileKey": file_key,
                        "code": _connectors_script(page_id, edges),
                        "description": f"Draw {len(edges)} transition connector(s) between screens",
                        "skillNames": "figma-use",
                    })

    figma_url = f"https://www.figma.com/design/{file_key}"
    log.info("Journey map published: %s", figma_url)
    return figma_url


def publish_to_figma(
    session: JourneySession,
    graph: nx.DiGraph,
    layout: dict[str, tuple[float, float]],
    page_name: str | None = None,
    progress_callback=None,
) -> str:
    """
    Publish the journey map to Figma via the remote MCP server.

    Returns the URL of the Figma file. On first use this opens a browser for a
    one-time OAuth authorization; subsequent calls reuse the cached token.
    """
    return asyncio.run(_publish_async(session, graph, layout, page_name, progress_callback))
