"""
Tests for capture agent helpers.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from autojourney.capture.agent import frames_from_file, save_frames


def _make_test_video(path: Path, num_frames: int = 10) -> None:
    """Write a minimal MP4 video with solid colour frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (100, 200))
    for i in range(num_frames):
        frame = np.full((200, 100, 3), i * 25 % 256, dtype=np.uint8)
        out.write(frame)
    out.release()


class TestFramesFromFile:
    def test_extracts_frames(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        _make_test_video(video_path, num_frames=10)
        frames = list(frames_from_file(video_path, fps_limit=10.0))
        assert len(frames) > 0

    def test_yields_tuples(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        _make_test_video(video_path, num_frames=5)
        for frame, index, ts in frames_from_file(video_path, fps_limit=10.0):
            assert isinstance(frame, np.ndarray)
            assert isinstance(index, int)
            assert isinstance(ts, int)
            break  # Just check first tuple

    def test_raises_on_missing_file(self):
        with pytest.raises(RuntimeError):
            list(frames_from_file(Path("nonexistent.mp4")))


class TestSaveFrames:
    def test_saves_pngs_and_manifest(self, tmp_path):
        def fake_source():
            for i in range(3):
                yield np.zeros((100, 50, 3), dtype=np.uint8), i, i * 100

        frames_dir, manifest = save_frames(fake_source(), output_dir=tmp_path)
        assert frames_dir.exists()
        assert len(manifest) == 3
        for entry in manifest:
            assert Path(entry["path"]).exists()

    def test_manifest_json_written(self, tmp_path):
        def fake_source():
            yield np.zeros((100, 50, 3), dtype=np.uint8), 0, 0

        frames_dir, manifest = save_frames(fake_source(), output_dir=tmp_path)
        manifest_path = frames_dir / "manifest.json"
        assert manifest_path.exists()
