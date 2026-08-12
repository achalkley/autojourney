"""
Tests for the Figma plugin bridge: the spec-building pure function and the
local HTTP server the plugin's UI iframe talks to.

The plugin's own JS (code.js / ui.html) runs inside Figma and isn't testable
here — same category as live USB capture.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from autojourney.figma.plugin_server import PluginBridgeServer
from autojourney.figma.publisher import FRAME_H, FRAME_W, _build_spec
from autojourney.models import EventType, FlowEdge, JourneySession, Screen


def _make_session(screens: list[Screen] | None = None, edges: list[FlowEdge] | None = None) -> JourneySession:
    if screens is None:
        screens = [
            Screen("s0001", Path("a.png"), 0, EventType.TRANSITION, app_name="MyApp", screen_name="Home"),
            Screen("s0002", Path("b.png"), 1000, EventType.TRANSITION, app_name="MyApp", screen_name="Detail"),
        ]
    if edges is None:
        edges = [FlowEdge("s0001", "s0002", "Tapped item", 1000)]
    return JourneySession("test_session", Path("video.mp4"), screens=screens, edges=edges)


class TestBuildSpec:
    def test_includes_file_key_and_page_name(self, monkeypatch):
        monkeypatch.setattr("autojourney.figma.publisher.config.FIGMA_FILE_KEY", "abc123")
        session = _make_session()
        spec = _build_spec(session, layout={}, page_name="MyPage")
        assert spec["fileKey"] == "abc123"
        assert spec["pageName"] == "MyPage"

    def test_falls_back_to_configured_page_name(self, monkeypatch):
        monkeypatch.setattr("autojourney.figma.publisher.config.FIGMA_PAGE_NAME", "DefaultPage")
        session = _make_session()
        spec = _build_spec(session, layout={}, page_name=None)
        assert spec["pageName"] == "DefaultPage"

    def test_layout_normalized_to_positive_offset(self):
        session = _make_session()
        layout = {"s0001": (-50.0, -20.0), "s0002": (100.0, 30.0)}
        spec = _build_spec(session, layout=layout, page_name="P")
        by_id = {s["id"]: s for s in spec["screens"]}
        assert by_id["s0001"]["x"] == pytest.approx(100.0)
        assert by_id["s0001"]["y"] == pytest.approx(100.0)
        assert by_id["s0002"]["x"] == pytest.approx(250.0)
        assert by_id["s0002"]["y"] == pytest.approx(150.0)

    def test_missing_layout_entry_falls_back_to_grid_position(self):
        session = _make_session()
        spec = _build_spec(session, layout={}, page_name="P")
        by_id = {s["id"]: s for s in spec["screens"]}
        assert by_id["s0001"]["x"] == 100
        assert by_id["s0002"]["x"] == 100 + FRAME_W + 120

    def test_frame_height_falls_back_to_default_for_unreadable_image(self):
        session = _make_session(screens=[
            Screen("s0001", Path("/nonexistent/a.png"), 0, EventType.TRANSITION),
        ])
        spec = _build_spec(session, layout={}, page_name="P")
        assert spec["screens"][0]["h"] == FRAME_H
        assert spec["screens"][0]["w"] == FRAME_W

    def test_frame_height_matches_image_aspect_ratio(self, tmp_path):
        img_path = tmp_path / "tall.png"
        # Native 200x800 (w x h) -> aspect ratio 4:1 tall
        image = np.zeros((800, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), image)
        session = _make_session(screens=[
            Screen("s0001", img_path, 0, EventType.TRANSITION),
        ])
        spec = _build_spec(session, layout={}, page_name="P")
        expected_h = int(FRAME_W * 800 / 200)
        assert spec["screens"][0]["h"] == expected_h

    def test_label_uses_app_and_screen_name(self):
        session = _make_session(screens=[
            Screen("s0001", Path("a.png"), 0, EventType.TRANSITION, app_name="ShopApp", screen_name="Home"),
        ])
        spec = _build_spec(session, layout={}, page_name="P")
        assert spec["screens"][0]["label"] == "ShopApp — Home"

    def test_label_falls_back_to_screen_id_when_no_metadata(self):
        session = _make_session(screens=[
            Screen("s0009", Path("a.png"), 0, EventType.TRANSITION),
        ])
        spec = _build_spec(session, layout={}, page_name="P")
        assert spec["screens"][0]["label"] == "Screen s0009"

    def test_action_is_none_when_not_inferred(self):
        session = _make_session(screens=[
            Screen("s0001", Path("a.png"), 0, EventType.TRANSITION),
        ])
        spec = _build_spec(session, layout={}, page_name="P")
        assert spec["screens"][0]["action"] is None

    def test_edges_included_when_both_endpoints_present(self):
        session = _make_session()
        spec = _build_spec(session, layout={}, page_name="P")
        assert spec["edges"] == [{"fromId": "s0001", "toId": "s0002", "label": "Tapped item"}]

    def test_dangling_edge_excluded(self):
        session = _make_session(edges=[FlowEdge("s0001", "s9999", "Ghost transition", 1000)])
        spec = _build_spec(session, layout={}, page_name="P")
        assert spec["edges"] == []


class TestPluginBridgeServer:
    def _start_server(self, spec, images, progress_callback=None) -> PluginBridgeServer:
        server = PluginBridgeServer(spec, images, progress_callback, port=0)
        server.start()
        return server

    def test_spec_endpoint_returns_json(self):
        spec = {"fileKey": "abc", "pageName": "P", "screens": [], "edges": []}
        server = self._start_server(spec, {})
        try:
            with urlopen(f"http://localhost:{server.port}/spec") as resp:
                assert resp.status == 200
                assert json.loads(resp.read()) == spec
        finally:
            server.stop()

    def test_image_endpoint_returns_bytes(self, tmp_path):
        img_path = tmp_path / "s.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
        server = self._start_server({"screens": []}, {"s0001": img_path})
        try:
            with urlopen(f"http://localhost:{server.port}/image/s0001") as resp:
                assert resp.status == 200
                assert resp.read() == img_path.read_bytes()
        finally:
            server.stop()

    def test_image_endpoint_404_for_unknown_screen(self):
        server = self._start_server({"screens": []}, {})
        try:
            with pytest.raises(HTTPError) as exc_info:
                urlopen(f"http://localhost:{server.port}/image/does-not-exist")
            assert exc_info.value.code == 404
        finally:
            server.stop()

    def test_progress_forwarded_to_callback(self):
        received = []
        server = self._start_server(
            {"screens": []}, {}, progress_callback=lambda done, total, detail: received.append((done, total, detail))
        )
        try:
            body = json.dumps({"done": 1, "total": 3, "detail": "Placed: Home"}).encode()
            req = Request(f"http://localhost:{server.port}/progress", data=body, method="POST",
                           headers={"Content-Type": "application/json"})
            with urlopen(req) as resp:
                assert resp.status == 204
            assert received == [(1, 3, "Placed: Home")]
        finally:
            server.stop()

    def test_complete_unblocks_wait_with_success(self):
        server = self._start_server({"screens": []}, {})

        def post_complete_soon():
            time.sleep(0.1)
            body = json.dumps({"success": True}).encode()
            req = Request(f"http://localhost:{server.port}/complete", data=body, method="POST",
                           headers={"Content-Type": "application/json"})
            urlopen(req).close()

        threading.Thread(target=post_complete_soon, daemon=True).start()
        server.wait_for_completion(timeout=5)  # must not raise

    def test_complete_with_failure_raises(self):
        server = self._start_server({"screens": []}, {})

        def post_complete_soon():
            time.sleep(0.1)
            body = json.dumps({"success": False, "error": "boom"}).encode()
            req = Request(f"http://localhost:{server.port}/complete", data=body, method="POST",
                           headers={"Content-Type": "application/json"})
            urlopen(req).close()

        threading.Thread(target=post_complete_soon, daemon=True).start()
        with pytest.raises(RuntimeError, match="boom"):
            server.wait_for_completion(timeout=5)

    def test_wait_times_out_when_plugin_never_runs(self):
        server = self._start_server({"screens": []}, {})
        with pytest.raises(TimeoutError):
            server.wait_for_completion(timeout=0.3)
