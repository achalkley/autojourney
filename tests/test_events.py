"""
Tests for event detection.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

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

@pytest.mark.xfail(
    strict=True,
    reason="P0-2: _ssim calls cv2.quality, which ships only in "
           "opencv-contrib-python. Under the declared opencv-python dependency "
           "every call raises AttributeError and is silently swallowed by ssim().",
)
def test_ssim_does_not_depend_on_contrib_only_cv2_quality():
    """
    `cv2.quality` is a contrib module. pyproject declares plain `opencv-python`,
    so the primary SSIM path can never run — the detector pays an exception on
    every frame pair and always lands in the manual fallback.
    """
    import inspect

    from autojourney.events import detector as detector_module

    source = inspect.getsource(detector_module)
    assert "cv2.quality" not in source, (
        "detector still references the contrib-only cv2.quality module"
    )


@pytest.mark.xfail(
    strict=True,
    reason="P1: __init__ uses `arg or config.DEFAULT`, so an explicit 0.0 is "
           "falsy and gets replaced by the config default.",
)
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
