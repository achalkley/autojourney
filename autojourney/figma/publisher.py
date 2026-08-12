"""
Figma integration via the official Figma MCP Server.

The Figma MCP Server (2025) exposes an HTTP API that allows programmatic
creation of nodes, upload of images, and placement of connectors.

This module talks to the MCP server via its HTTP transport.

Reference: https://developers.figma.com/docs/figma-mcp-server/
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import networkx as nx

from autojourney import config
from autojourney.models import JourneySession

log = logging.getLogger(__name__)

# Figma REST API base (used for file/page creation)
FIGMA_API_BASE = "https://api.figma.com/v1"

# Layout constants (Figma units ≈ pixels at 1x)
FRAME_W = 390        # iPhone 15 logical width
FRAME_H = 844        # iPhone 15 logical height
H_GAP = 120          # Horizontal gap between frames
V_GAP = 200          # Vertical gap between tree levels
LABEL_H = 60         # Height below frame for screen name label
CONNECTOR_LABEL_OFFSET = 20


class FigmaMCPClient:
    """
    Thin wrapper around the Figma MCP Server HTTP transport.

    The MCP server exposes a /mcp endpoint that accepts JSON-RPC 2.0 style
    tool invocations. We call the documented tools:
      - create_frame
      - upload_image_to_frame
      - create_text
      - create_connector
    """

    def __init__(self, mcp_base_url: str = "http://localhost:3000") -> None:
        self.base = mcp_base_url.rstrip("/")
        self.token = config.FIGMA_API_TOKEN
        self.file_key = config.FIGMA_FILE_KEY
        self._headers = {
            "Content-Type": "application/json",
            "X-Figma-Token": self.token,
        }

    def _call(self, tool: str, params: dict[str, Any]) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": params},
        }
        resp = httpx.post(f"{self.base}/mcp", json=payload, headers=self._headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            raise RuntimeError(f"MCP error [{tool}]: {result['error']}")
        return result.get("result", {})

    def ensure_page(self, page_name: str) -> str:
        """Create or retrieve the journey map page. Returns page node ID."""
        result = self._call("get_file", {"file_key": self.file_key})
        pages = result.get("document", {}).get("children", [])
        for page in pages:
            if page.get("name") == page_name:
                log.info("Using existing Figma page: %s (%s)", page_name, page["id"])
                return page["id"]

        result = self._call("create_page", {
            "file_key": self.file_key,
            "name": page_name,
        })
        page_id = result["id"]
        log.info("Created Figma page: %s (%s)", page_name, page_id)
        return page_id

    def create_frame(
        self,
        page_id: str,
        name: str,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> str:
        """Create a frame and return its node ID."""
        result = self._call("create_frame", {
            "file_key": self.file_key,
            "page_id": page_id,
            "name": name,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        })
        return result["id"]

    def upload_image(self, frame_id: str, image_path: Path) -> None:
        """Upload a PNG/JPG as the fill image for a frame."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        self._call("upload_image_to_frame", {
            "file_key": self.file_key,
            "frame_id": frame_id,
            "image_data": b64,
            "mime_type": "image/png",
        })

    def create_text(
        self,
        page_id: str,
        text: str,
        x: float,
        y: float,
        width: float,
        font_size: int = 14,
        bold: bool = False,
    ) -> str:
        result = self._call("create_text", {
            "file_key": self.file_key,
            "page_id": page_id,
            "content": text,
            "x": x,
            "y": y,
            "width": width,
            "font_size": font_size,
            "font_weight": 700 if bold else 400,
        })
        return result["id"]

    def create_connector(
        self,
        page_id: str,
        from_node_id: str,
        to_node_id: str,
        label: str = "",
    ) -> str:
        result = self._call("create_connector", {
            "file_key": self.file_key,
            "page_id": page_id,
            "start_node_id": from_node_id,
            "end_node_id": to_node_id,
            "label": label,
        })
        return result["id"]


# ──────────────────────────────────────────────────────────────────────────────
# Main publish function
# ──────────────────────────────────────────────────────────────────────────────

def publish_to_figma(
    session: JourneySession,
    graph: nx.DiGraph,
    layout: dict[str, tuple[float, float]],
    mcp_base_url: str = "http://localhost:3000",
    page_name: str | None = None,
    progress_callback=None,
) -> str:
    """
    Publish the journey map to Figma via the MCP server.

    Returns the URL of the Figma file.
    """
    client = FigmaMCPClient(mcp_base_url=mcp_base_url)
    target_page = page_name or config.FIGMA_PAGE_NAME

    log.info("Publishing journey map to Figma (file: %s, page: %s)", config.FIGMA_FILE_KEY, target_page)
    page_id = client.ensure_page(target_page)

    # Normalise layout coordinates so the origin is at (100, 100)
    if layout:
        min_x = min(x for x, _ in layout.values())
        min_y = min(y for _, y in layout.values())
        layout = {n: (x - min_x + 100, y - min_y + 100) for n, (x, y) in layout.items()}

    # Map screen_id → figma frame node ID
    node_map: dict[str, str] = {}
    screen_map = {s.screen_id: s for s in session.screens}

    total = len(session.screens)
    for i, screen in enumerate(session.screens):
        lx, ly = layout.get(screen.screen_id, (i * (FRAME_W + H_GAP) + 100, 100))

        # Determine frame height (tall for stitched scroll images)
        try:
            import cv2
            img = cv2.imread(str(screen.image_path))
            if img is not None:
                native_h, native_w = img.shape[:2]
                frame_h = int(FRAME_W * native_h / native_w)
            else:
                frame_h = FRAME_H
        except Exception:
            frame_h = FRAME_H

        frame_label = screen.screen_name or f"Screen {screen.screen_id}"
        frame_id = client.create_frame(
            page_id=page_id,
            name=frame_label,
            x=lx,
            y=ly,
            width=FRAME_W,
            height=frame_h,
        )
        node_map[screen.screen_id] = frame_id

        # Upload screenshot
        client.upload_image(frame_id, screen.image_path)

        # Screen name label below frame
        client.create_text(
            page_id=page_id,
            text=f"{screen.app_name} — {screen.screen_name}" if screen.app_name else frame_label,
            x=lx,
            y=ly + frame_h + 8,
            width=FRAME_W,
            font_size=13,
            bold=True,
        )

        # Inferred action label (smaller, italic-style)
        if screen.inferred_action:
            client.create_text(
                page_id=page_id,
                text=f"↳ {screen.inferred_action}",
                x=lx,
                y=ly + frame_h + 28,
                width=FRAME_W,
                font_size=11,
            )

        if progress_callback:
            progress_callback(i + 1, total, f"Placed: {frame_label}")

    # Draw connectors
    for edge in session.edges:
        if edge.from_screen_id in node_map and edge.to_screen_id in node_map:
            client.create_connector(
                page_id=page_id,
                from_node_id=node_map[edge.from_screen_id],
                to_node_id=node_map[edge.to_screen_id],
                label=edge.interaction_label,
            )

    figma_url = f"https://www.figma.com/file/{config.FIGMA_FILE_KEY}"
    log.info("Journey map published: %s", figma_url)
    return figma_url
