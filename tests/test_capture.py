"""
Tests for capture agent helpers.
"""
from __future__ import annotations

import json
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


# ──────────────────────────────────────────────────────────────────────────────
# Regression coverage for known defects (audit Phase 2 / P0-4)
# ──────────────────────────────────────────────────────────────────────────────

class TestSaveFramesInterrupted:
    """
    A KeyboardInterrupt used to fire inside the frame-writing loop and
    propagate straight out of save_frames, skipping the manifest.json write
    entirely — frames already on disk with no record of what they were,
    unrecoverable by the rest of the pipeline.
    """

    def _interrupted_source(self, n_before_interrupt: int):
        def fake_source():
            for i in range(n_before_interrupt):
                yield np.zeros((100, 50, 3), dtype=np.uint8), i, i * 100
            raise KeyboardInterrupt

        return fake_source()

    def test_frames_captured_before_interrupt_are_persisted(self, tmp_path):
        """swallow_interrupt=True: the live-USB case — Ctrl+C means 'stop
        capturing', not 'abort'. save_frames must return normally with
        whatever was captured, not raise."""
        frames_dir, manifest = save_frames(
            self._interrupted_source(3), output_dir=tmp_path, swallow_interrupt=True
        )
        assert len(manifest) == 3
        for entry in manifest:
            assert Path(entry["path"]).exists()

        manifest_path = frames_dir / "manifest.json"
        assert manifest_path.exists()
        on_disk = json.loads(manifest_path.read_text())
        assert len(on_disk) == 3

    def test_interrupt_still_raises_when_not_swallowed(self, tmp_path):
        """swallow_interrupt=False (the default): file-based capture is a
        batch job — Ctrl+C should still abort it, just without losing the
        frames already written."""
        with pytest.raises(KeyboardInterrupt):
            save_frames(self._interrupted_source(2), output_dir=tmp_path)

    def test_manifest_persisted_even_when_interrupt_propagates(self, tmp_path):
        try:
            save_frames(self._interrupted_source(2), output_dir=tmp_path)
        except KeyboardInterrupt:
            pass

        manifest_path = tmp_path / "frames" / "manifest.json"
        assert manifest_path.exists()
        on_disk = json.loads(manifest_path.read_text())
        assert len(on_disk) == 2

    def test_interrupt_before_any_frame_yields_empty_manifest_not_a_crash(self, tmp_path):
        def fake_source():
            raise KeyboardInterrupt
            yield  # pragma: no cover — unreachable, makes this a generator

        frames_dir, manifest = save_frames(fake_source(), output_dir=tmp_path, swallow_interrupt=True)
        assert manifest == []
        assert (frames_dir / "manifest.json").exists()
