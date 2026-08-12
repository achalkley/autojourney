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
