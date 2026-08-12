"""
Tests for event detection.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from autojourney.events.detector import (
    EventDetector,
    changed_area_fraction,
    dominant_flow,
    is_modal,
    ssim,
)
from autojourney.models import EventType


def _solid_frame(color: tuple[int, int, int], h: int = 200, w: int = 100) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _save_frame(frame: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), frame)


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — image analysis helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestSSIM:
    def test_identical_frames_score_1(self):
        frame = _solid_frame((100, 150, 200))
        score = ssim(frame, frame.copy())
        assert score > 0.99

    def test_different_frames_score_low(self):
        a = _solid_frame((0, 0, 0))
        b = _solid_frame((255, 255, 255))
        score = ssim(a, b)
        assert score < 0.5

    def test_handles_different_sizes(self):
        a = _solid_frame((100, 100, 100), h=200, w=100)
        b = _solid_frame((100, 100, 100), h=400, w=200)
        score = ssim(a, b)
        assert score > 0.90


class TestChangedAreaFraction:
    def test_identical(self):
        frame = _solid_frame((100, 100, 100))
        assert changed_area_fraction(frame, frame.copy()) == 0.0

    def test_half_changed(self):
        a = _solid_frame((0, 0, 0), h=100, w=100)
        b = a.copy()
        b[50:, :] = 255  # bottom half changed
        frac = changed_area_fraction(a, b)
        assert 0.45 < frac < 0.55

    def test_fully_different(self):
        a = _solid_frame((0, 0, 0))
        b = _solid_frame((255, 255, 255))
        frac = changed_area_fraction(a, b)
        assert frac > 0.99


class TestIsModal:
    def test_stable_top_changing_bottom_is_modal(self):
        a = _solid_frame((100, 100, 100), h=400, w=200)
        b = a.copy()
        # Keep top quarter identical, change bottom 60%
        b[100:, :] = [200, 50, 50]
        assert is_modal(a, b) is True

    def test_full_change_not_modal(self):
        a = _solid_frame((0, 0, 0), h=400, w=200)
        b = _solid_frame((200, 200, 200), h=400, w=200)
        assert is_modal(a, b) is False


# ──────────────────────────────────────────────────────────────────────────────
# Integration test — EventDetector with a manifest
# ──────────────────────────────────────────────────────────────────────────────

class TestEventDetector:
    def _build_manifest(self, frames: list[np.ndarray], tmp_path: Path) -> list[dict]:
        manifest = []
        for i, frame in enumerate(frames):
            p = tmp_path / f"frame_{i:06d}.png"
            _save_frame(frame, p)
            manifest.append({"index": i, "timestamp_ms": i * 200, "path": str(p)})
        return manifest

    def test_detects_transition(self, tmp_path):
        frames = [_solid_frame((0, 0, 0)) for _ in range(3)]
        frames += [_solid_frame((200, 200, 200)) for _ in range(3)]  # transition here
        manifest = self._build_manifest(frames, tmp_path)
        detector = EventDetector(manifest, tmp_path, transition_ssim=0.85, transition_area=0.1)
        events = list(detector.detect())
        types = [e.event_type for e in events]
        assert EventType.TRANSITION in types

    def test_video_end_always_emitted(self, tmp_path):
        frames = [_solid_frame((100, 100, 100)) for _ in range(4)]
        manifest = self._build_manifest(frames, tmp_path)
        detector = EventDetector(manifest, tmp_path)
        events = list(detector.detect())
        assert events[-1].event_type == EventType.VIDEO_END

    def test_empty_manifest_yields_nothing(self, tmp_path):
        detector = EventDetector([], tmp_path)
        events = list(detector.detect())
        assert events == []


# ──────────────────────────────────────────────────────────────────────────────
# Regression coverage for known defects (audit P0-2, P1 thresholds)
# ──────────────────────────────────────────────────────────────────────────────

def test_ssim_correct_without_contrib_module(monkeypatch):
    """
    SSIM must be correct under the declared `opencv-python` dependency, which
    does not ship `cv2.quality` (that module is contrib-only). ssim() no
    longer references it at all (P0-2); delattr here is a cheap guard against
    that reference coming back.
    """
    monkeypatch.delattr(cv2, "quality", raising=False)

    identical = _solid_frame((100, 150, 200))
    assert ssim(identical, identical.copy()) > 0.99

    dark = _solid_frame((0, 0, 0))
    light = _solid_frame((255, 255, 255))
    assert ssim(dark, light) < 0.5


def test_explicit_zero_thresholds_are_respected(tmp_path):
    """0.0 is a legitimate threshold (disable the check), not 'unset'."""
    detector = EventDetector(
        [],
        tmp_path,
        transition_ssim=0.0,
        transition_area=0.0,
        scroll_flow=0.0,
    )
    assert detector.transition_ssim == 0.0
    assert detector.transition_area == 0.0
    assert detector.scroll_flow == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Regression coverage for known defects (audit Phase 2: frame-read resilience)
# ──────────────────────────────────────────────────────────────────────────────

class TestUnreadableFrames:
    def _build_manifest(self, frames: list[np.ndarray], tmp_path: Path) -> list[dict]:
        manifest = []
        for i, frame in enumerate(frames):
            p = tmp_path / f"frame_{i:06d}.png"
            _save_frame(frame, p)
            manifest.append({"index": i, "timestamp_ms": i * 200, "path": str(p)})
        return manifest

    def test_corrupt_middle_frame_does_not_abort_the_run(self, tmp_path):
        """
        A single unreadable frame anywhere in the manifest used to raise
        RuntimeError out of detect(), killing every event after it. It should
        instead be skipped, as if that frame were never captured.
        """
        frames = [_solid_frame((0, 0, 0)) for _ in range(3)]
        frames += [_solid_frame((200, 200, 200)) for _ in range(3)]  # transition here
        manifest = self._build_manifest(frames, tmp_path)
        # Point one middle entry at a path that doesn't decode as an image.
        bogus = tmp_path / "corrupt.png"
        bogus.write_bytes(b"not a png")
        manifest[2]["path"] = str(bogus)

        detector = EventDetector(manifest, tmp_path, transition_ssim=0.85, transition_area=0.1)
        events = list(detector.detect())  # must not raise

        types = [e.event_type for e in events]
        assert EventType.TRANSITION in types
        assert events[-1].event_type == EventType.VIDEO_END

    def test_corrupt_first_frame_does_not_abort_the_run(self, tmp_path):
        """The very first manifest entry being unreadable is the sharpest case:
        the old code loaded it unconditionally before the loop even started."""
        frames = [_solid_frame((0, 0, 0)) for _ in range(2)]
        frames += [_solid_frame((200, 200, 200)) for _ in range(2)]
        manifest = self._build_manifest(frames, tmp_path)
        bogus = tmp_path / "corrupt.png"
        bogus.write_bytes(b"not a png")
        manifest[0]["path"] = str(bogus)

        detector = EventDetector(manifest, tmp_path, transition_ssim=0.85, transition_area=0.1)
        events = list(detector.detect())  # must not raise
        assert events[-1].event_type == EventType.VIDEO_END

    def test_all_frames_unreadable_yields_only_video_end(self, tmp_path):
        manifest = [{"index": 0, "timestamp_ms": 0, "path": str(tmp_path / "missing.png")}]
        detector = EventDetector(manifest, tmp_path)
        events = list(detector.detect())  # must not raise
        assert len(events) == 1
        assert events[0].event_type == EventType.VIDEO_END
