"""
End-to-end pipeline tests against a synthetic video.

The LLM analyser is stubbed out — these tests cover the wiring between
capture, event detection, screen collection and reporting, not the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autojourney import pipeline as pipeline_module
from autojourney.pipeline import run_pipeline

NATIVE_FPS = 10.0
FRAMES_PER_SCENE = 10


def _two_scene_video(path: Path) -> None:
    """
    Write a video with one hard cut in the middle: 10 dark frames followed by
    10 light frames. That is exactly one screen transition, so a correct
    pipeline yields two screens — the starting screen and the one after the cut.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, NATIVE_FPS, (100, 200))
    if not out.isOpened():
        pytest.skip("OpenCV build has no mp4v encoder — cannot build the fixture")
    for value in (20, 230):
        frame = np.full((200, 100, 3), value, dtype=np.uint8)
        for _ in range(FRAMES_PER_SCENE):
            out.write(frame)
    out.release()


@pytest.fixture
def stub_analyser(monkeypatch):
    """Replace the network-bound LLM call with a deterministic stub."""
    calls: list[Path] = []

    def _fake_analyse(image_path: Path) -> dict:
        calls.append(image_path)
        return {
            "app_name": "TestApp",
            "screen_name": f"Screen {len(calls)}",
            "ui_elements": ["Button"],
            "inferred_action": "Tapped something",
            "probable_destinations": ["Next"],
        }

    monkeypatch.setattr(pipeline_module, "analyse_screen", _fake_analyse)
    return calls


def _run(tmp_path: Path):
    video = tmp_path / "session.mp4"
    _two_scene_video(video)
    return run_pipeline(
        video_path=video,
        output_dir=tmp_path / "out",
        publish=False,
        fps_limit=NATIVE_FPS,
    )


class TestPipelineSmoke:
    def test_produces_screens_and_outputs(self, tmp_path, stub_analyser):
        session = _run(tmp_path)
        out = tmp_path / "out"

        assert session.screens, "pipeline produced no screens"
        assert (out / "frames" / "manifest.json").exists()
        assert (out / "events.json").exists()
        assert (out / "session.json").exists()
        assert (out / "journey-report.md").exists()

    def test_session_json_round_trips(self, tmp_path, stub_analyser):
        session = _run(tmp_path)
        raw = json.loads((tmp_path / "out" / "session.json").read_text())
        assert len(raw["screens"]) == len(session.screens)
        assert len(raw["edges"]) == len(session.edges)

    def test_edges_reference_known_screens(self, tmp_path, stub_analyser):
        session = _run(tmp_path)
        assert session.edges, "pipeline produced no edges"
        ids = {s.screen_id for s in session.screens}
        for edge in session.edges:
            assert edge.from_screen_id in ids
            assert edge.to_screen_id in ids


# ──────────────────────────────────────────────────────────────────────────────
# Regression coverage for known defects (audit P0-3)
# ──────────────────────────────────────────────────────────────────────────────

def test_captures_the_screen_the_journey_starts_on(tmp_path, stub_analyser):
    """
    The video opens on a dark screen and cuts to a light one. Both are part of
    the journey — the starting screen is where the user began, and a journey map
    that omits it is missing its root node.
    """
    session = _run(tmp_path)

    assert len(session.screens) >= 2, (
        f"expected the starting screen plus the post-transition screen, "
        f"got {len(session.screens)}"
    )
    first = session.screens[0]
    assert first.timestamp_ms == 0, (
        f"first screen is at {first.timestamp_ms}ms — the journey's opening "
        f"screen was never captured"
    )

    opening = cv2.imread(str(first.image_path))
    assert opening is not None
    assert opening.mean() < 128, (
        "first screen is the post-transition light frame, not the dark frame "
        "the session opened on"
    )

    # With both screens present there is a transition between them to record.
    assert session.edges, "no edge recorded for the cut between the two screens"


# ──────────────────────────────────────────────────────────────────────────────
# Regression coverage for known defects (audit Phase 2 / P0-4)
# ──────────────────────────────────────────────────────────────────────────────

def test_ctrl_c_during_usb_capture_continues_through_the_pipeline(tmp_path, stub_analyser, monkeypatch):
    """
    Ctrl+C during live USB capture used to propagate a raw KeyboardInterrupt
    out of save_frames before manifest.json was written, aborting the whole
    run with frames on disk but no way to use them. It's also the documented
    way to end a capture session (README: "Press Ctrl+C when done
    navigating"), so the pipeline should treat it as "stop capturing" and
    keep going through event detection, analysis and the report — not as an
    abort.
    """
    import autojourney.capture.agent as agent_module

    def fake_frames_from_usb(fps_limit: float = 5.0, stop_event=None):
        frame_a = np.full((200, 100, 3), 20, dtype=np.uint8)
        frame_b = np.full((200, 100, 3), 230, dtype=np.uint8)
        for i in range(FRAMES_PER_SCENE):
            yield frame_a, i, i * 100
        for i in range(FRAMES_PER_SCENE, 2 * FRAMES_PER_SCENE - 2):
            yield frame_b, i, i * 100
        raise KeyboardInterrupt  # user pressed Ctrl+C

    monkeypatch.setattr(agent_module, "frames_from_usb", fake_frames_from_usb)

    session = run_pipeline(
        video_path=None,  # selects the USB capture branch
        output_dir=tmp_path / "out",
        publish=False,
        fps_limit=NATIVE_FPS,
    )  # must not raise

    out = tmp_path / "out"
    assert (out / "frames" / "manifest.json").exists()
    manifest = json.loads((out / "frames" / "manifest.json").read_text())
    assert len(manifest) == 2 * FRAMES_PER_SCENE - 2, "frames before the interrupt were not all persisted"

    # The interrupt must not have short-circuited the rest of the pipeline.
    assert session.screens, "pipeline stopped instead of continuing past the interrupt"
    assert (out / "events.json").exists()
    assert (out / "session.json").exists()
    assert (out / "journey-report.md").exists()
