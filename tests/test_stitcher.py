"""
Tests for scroll stitcher.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from autojourney.stitcher.scroll import stitch_scroll, _find_overlap_offset


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


def _tall_page() -> np.ndarray:
    """
    A tall page of smooth, non-repeating texture.

    Upscaled low-resolution noise, deliberately: per-pixel noise has no
    structure for Farneback optical flow to track, so `_detect_scroll_direction`
    misreads it as horizontal, and a periodic gradient gives template matching
    several equally good alignments. This gives one unambiguous match per pair
    and a flow field that reads as a vertical scroll.
    """
    rng = np.random.default_rng(7)
    small = rng.integers(0, 256, size=(PAGE_H // 10, FRAME_W // 10, 3), dtype=np.uint8)
    return cv2.resize(small, (FRAME_W, PAGE_H), interpolation=cv2.INTER_CUBIC)


def _scroll_sequence(tmp_path: Path) -> tuple[list[Path], int]:
    """
    Simulate a viewport scrolling down a tall page.

    Returns (frame_paths, expected_stitched_height).
    """
    page = _tall_page()
    paths = []
    for i in range(NUM_FRAMES):
        top = i * SCROLL_STEP
        p = tmp_path / f"scroll_{i:06d}.png"
        cv2.imwrite(str(p), page[top:top + FRAME_H])
        paths.append(p)
    return paths, PAGE_H


@pytest.mark.xfail(
    strict=True,
    reason="P0-1: _find_overlap_offset derives geometry from the growing composite "
           "instead of the previous frame, so overlap is never removed and the "
           "stitch degenerates into a plain vstack of every frame.",
)
def test_stitch_removes_overlap_across_long_scroll(tmp_path):
    """A 12-frame scroll over a 1400px page must produce ~1400px, not ~3200px."""
    paths, expected_h = _scroll_sequence(tmp_path)
    result = stitch_scroll(paths)

    naive_concat_h = FRAME_H * NUM_FRAMES
    assert result.shape[0] < naive_concat_h, (
        f"stitched height {result.shape[0]} is no better than concatenating "
        f"every frame ({naive_concat_h}) — no overlap was removed"
    )
    assert expected_h * 0.9 <= result.shape[0] <= expected_h * 1.1, (
        f"stitched height {result.shape[0]} is not within 10% of the true page "
        f"height {expected_h}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="P0-1: the offset is computed against the composite's height, so it "
           "exceeds frame_b's height and the negative slice returns the whole frame.",
)
def test_overlap_offset_never_exceeds_source_frame_height(tmp_path):
    """
    The returned offset is a count of rows to take *from frame_b*, so it can
    never exceed frame_b's height — no matter how tall the accumulated
    composite passed as frame_a has grown.
    """
    page = _tall_page()
    composite = np.vstack([page, page])   # stands in for an accumulated stitch
    frame_b = page[-FRAME_H:].copy()

    unique = _find_overlap_offset(composite, frame_b, direction="vertical")
    assert unique <= frame_b.shape[0], (
        f"offset {unique} exceeds frame_b height {frame_b.shape[0]}"
    )
