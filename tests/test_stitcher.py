"""
Tests for scroll stitcher.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from autojourney.stitcher.scroll import (
    stitch_scroll,
    _detect_scroll_direction,
    _find_overlap_offset,
)


def _gradient_frame(start_row: int, h: int = 300, w: int = 200) -> np.ndarray:
    """Create a frame with a vertical gradient shifted by start_row."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        val = (start_row + row) % 256
        frame[row, :] = [val, val, val]
    return frame


class TestFindOverlapOffset:
    def test_vertical_overlap(self, tmp_path):
        a = _gradient_frame(0)    # rows 0–299
        b = _gradient_frame(200)  # rows 200–499 — overlaps bottom 100 rows of a
        unique = _find_overlap_offset(a, b, direction="vertical")
        # b extends 200 unique rows below a
        assert unique > 0
        assert unique <= a.shape[0]


class TestStitchScroll:
    def test_single_frame_returned_as_is(self, tmp_path):
        frame = _gradient_frame(0)
        p = tmp_path / "frame_000000.png"
        cv2.imwrite(str(p), frame)
        result = stitch_scroll([p])
        assert result.shape == frame.shape

    def test_two_frames_produce_taller_image(self, tmp_path):
        paths = []
        for i in range(3):
            frame = _gradient_frame(i * 200)
            p = tmp_path / f"frame_{i:06d}.png"
            cv2.imwrite(str(p), frame)
            paths.append(p)

        result = stitch_scroll(paths)
        # Should be taller than a single frame (some overlap removed)
        assert result.shape[0] > 300

    def test_output_file_written(self, tmp_path):
        frame = _gradient_frame(0)
        p = tmp_path / "frame_000000.png"
        cv2.imwrite(str(p), frame)
        out = tmp_path / "stitched.png"
        stitch_scroll([p], output_path=out)
        assert out.exists()

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            stitch_scroll([])


# ──────────────────────────────────────────────────────────────────────────────
# Regression coverage for known defects (audit P0-1)
# ──────────────────────────────────────────────────────────────────────────────

FRAME_H = 300
FRAME_W = 200
SCROLL_STEP = 100
NUM_FRAMES = 12
PAGE_H = SCROLL_STEP * (NUM_FRAMES - 1) + FRAME_H


def _tall_page(page_h: int = PAGE_H, seed: int = 7) -> np.ndarray:
    """
    A tall page of smooth, non-repeating texture: upscaled low-resolution noise.

    A safe general-purpose choice for exercising template matching over a
    scroll sequence — unlike a periodic gradient it has no repeat interval
    for a large search window to alias onto, and unlike raw per-pixel noise
    it still resembles the kind of soft-edged UI content the stitcher is
    built for.
    """
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(page_h // 10, FRAME_W // 10, 3), dtype=np.uint8)
    return cv2.resize(small, (FRAME_W, page_h), interpolation=cv2.INTER_CUBIC)


def _scroll_sequence(
    tmp_path: Path, num_frames: int = NUM_FRAMES, step: int = SCROLL_STEP
) -> tuple[list[Path], int]:
    """
    Simulate a viewport scrolling down a tall page.

    Returns (frame_paths, expected_stitched_height).
    """
    page_h = step * (num_frames - 1) + FRAME_H
    page = _tall_page(page_h)
    paths = []
    for i in range(num_frames):
        top = i * step
        p = tmp_path / f"scroll_{i:06d}.png"
        cv2.imwrite(str(p), page[top:top + FRAME_H])
        paths.append(p)
    return paths, page_h


def test_stitch_removes_overlap_across_long_scroll(tmp_path):
    """A 12-frame scroll over a 1400px page must produce ~1400px, not ~3000px."""
    paths, expected_h = _scroll_sequence(tmp_path)
    result = stitch_scroll(paths)

    assert expected_h * 0.9 <= result.shape[0] <= expected_h * 1.1, (
        f"stitched height {result.shape[0]} is not within 10% of the true page "
        f"height {expected_h}"
    )


def test_direction_detection_stable_over_long_scroll(tmp_path):
    """
    Comparing only the first and last frame of a scroll (the original
    implementation) loses the plot once a sequence runs long enough that
    those two frames no longer overlap at all — confirmed empirically to
    misclassify this exact fixture as horizontal at 30+ frames. Direction
    must stay correct regardless of how long the scroll ran.
    """
    paths, _ = _scroll_sequence(tmp_path, num_frames=30)
    frames = [cv2.imread(str(p)) for p in paths]
    assert _detect_scroll_direction(frames) == "vertical"
