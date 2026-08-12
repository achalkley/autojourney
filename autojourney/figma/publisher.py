"""
Figma integration via a local Figma Plugin.

Figma's remote MCP server requires OAuth dynamic client registration, which
is gated to clients listed in Figma's MCP Catalog — third-party apps like
AutoJourney can't register. The local desktop MCP server has no such gate,
but doesn't expose `use_figma`/`upload_assets`, the only tools that can write
canvas content. Neither works here.

Instead, this module drives an actual Figma Plugin (`autojourney/figma_plugin/`)
running inside the user's own Figma session — no OAuth, no catalog. The
plugin's main sandbox has document/scene-graph access but no network access
(a Plugin API restriction); its UI iframe has network access but no document
access. So this module starts a local HTTP bridge server that the iframe
fetches the session spec and screen images from, and the plugin does the
actual node creation with `figma.createImage()` for placing screenshots.

See the README's "Figma publishing" section for the one-time plugin import
step.
"""
from __future__ import annotations

import logging
import webbrowser
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

from autojourney import config
from autojourney.figma.plugin_server import PluginBridgeServer
from autojourney.models import JourneySession

log = logging.getLogger(__name__)

# Layout constants (Figma units ≈ pixels at 1x)
FRAME_W = 390        # iPhone 15 logical width
FRAME_H = 844        # iPhone 15 logical height
H_GAP = 120           # Horizontal gap between frames


def _build_spec(
    session: JourneySession,
    layout: dict[str, tuple[float, float]],
    page_name: str | None,
) -> dict[str, Any]:
    """Build the plain-JSON spec the plugin's UI iframe fetches from /spec."""
    target_page = page_name or config.FIGMA_PAGE_NAME

    if layout:
        min_x = min(x for x, _ in layout.values())
        min_y = min(y for _, y in layout.values())
        layout = {n: (x - min_x + 100, y - min_y + 100) for n, (x, y) in layout.items()}

    screens = []
    for i, screen in enumerate(session.screens):
        lx, ly = layout.get(screen.screen_id, (i * (FRAME_W + H_GAP) + 100, 100))
        frame_h = FRAME_H
        try:
            import cv2
            img = cv2.imread(str(screen.image_path))
            if img is not None:
                native_h, native_w = img.shape[:2]
                frame_h = int(FRAME_W * native_h / native_w)
        except Exception:  # noqa: BLE001, S110 — best-effort aspect-ratio read; fall back to the default frame height
            pass
        frame_label = screen.screen_name or f"Screen {screen.screen_id}"
        screens.append({
            "id": screen.screen_id,
            "name": frame_label,
            "x": lx,
            "y": ly,
            "w": FRAME_W,
            "h": frame_h,
            "label": f"{screen.app_name} — {screen.screen_name}" if screen.app_name else frame_label,
            "action": screen.inferred_action or None,
        })

    screen_ids = {s["id"] for s in screens}
    edges = [
        {"fromId": e.from_screen_id, "toId": e.to_screen_id, "label": e.interaction_label}
        for e in session.edges
        if e.from_screen_id in screen_ids and e.to_screen_id in screen_ids
    ]

    return {
        "fileKey": config.FIGMA_FILE_KEY,
        "pageName": target_page,
        "screens": screens,
        "edges": edges,
    }


def publish_to_figma(
    session: JourneySession,
    layout: dict[str, tuple[float, float]],
    page_name: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> str:
    """
    Publish the journey map to Figma via the AutoJourney Figma plugin.

    Starts a local bridge server and waits for the plugin to run inside the
    target Figma file (Plugins → Development → AutoJourney). Returns the
    Figma file URL once the plugin reports completion.
    """
    spec = _build_spec(session, layout, page_name)
    images = {s.screen_id: Path(s.image_path) for s in session.screens}

    server = PluginBridgeServer(spec, images, progress_callback)
    server.start()

    file_key = config.FIGMA_FILE_KEY
    figma_url = f"https://www.figma.com/design/{file_key}"
    manifest_path = resources.files("autojourney.figma_plugin") / "manifest.json"

    log.info("Bridge server running on http://localhost:%d", server.port)
    log.info(
        "If you haven't already, import the AutoJourney plugin once: in the Figma "
        "desktop app go to Plugins → Development → Import plugin from manifest…"
    )
    log.info("  %s", manifest_path)
    log.info("Then open the file and run Plugins → Development → AutoJourney: %s", figma_url)
    webbrowser.open(figma_url)

    server.wait_for_completion()
    log.info("Journey map published: %s", figma_url)
    return figma_url
