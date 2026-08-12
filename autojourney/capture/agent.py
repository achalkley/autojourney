"""
Capture agent — extracts frames from a source.

Supported sources:
  1. Pre-recorded .mp4 / .mov file (--source path/to/video.mp4)
  2. Live USB stream from a connected iOS device via the QuickTime
     protocol, using the `ios-screen-record` package.

Frames are written to OUTPUT_DIR/frames/ as PNG files named
by their zero-padded frame index, e.g. frame_000123.png.
A manifest JSON is written alongside them.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from autojourney import config

log = logging.getLogger(__name__)

FrameCallback = Callable[[np.ndarray, int, int], None]  # frame, index, timestamp_ms


# ──────────────────────────────────────────────────────────────────────────────
# File-based capture
# ──────────────────────────────────────────────────────────────────────────────

def frames_from_file(
    video_path: Path,
    fps_limit: float = 5.0,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """
    Yield (frame_bgr, frame_index, timestamp_ms) from a video file.

    fps_limit: maximum frames per second to extract (reduces processing load).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / fps_limit))
    frame_index = 0
    extracted = 0

    log.info("Video: %.1f fps native, extracting every %d frames (≈%.1f fps)", native_fps, step, native_fps / step)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % step == 0:
                timestamp_ms = int((frame_index / native_fps) * 1000)
                yield frame, extracted, timestamp_ms
                extracted += 1
            frame_index += 1
    finally:
        cap.release()

    log.info("Extracted %d frames from %s", extracted, video_path)


# ──────────────────────────────────────────────────────────────────────────────
# Live USB capture via ios-screen-record
# ──────────────────────────────────────────────────────────────────────────────

def frames_from_usb(
    fps_limit: float = 5.0,
    stop_event: threading.Event | None = None,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """
    Yield (frame_bgr, frame_index, timestamp_ms) from a live iOS device
    connected over USB.

    Requires `ios-screen-record` to be installed:
        pip install ios-screen-record

    The library exposes an MJPEG/H.264 stream which we pipe through FFmpeg
    into OpenCV via stdout.

    Raises ImportError if ios-screen-record is not installed.
    Raises RuntimeError if no device is detected.
    """
    try:
        import iosscreenrecord  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "ios-screen-record is required for USB capture. "
            "Install it with: pip install 'autojourney[capture]'"
        ) from exc

    log.info("Starting USB capture from connected iOS device …")

    # ios-screen-record provides a context manager that yields raw H.264 frames
    # via a queue. We convert each to BGR using cv2.imdecode.
    start_time = time.time()
    frame_index = 0
    last_yielded = 0.0
    min_interval = 1.0 / fps_limit

    with iosscreenrecord.record() as recorder:
        for raw_frame in recorder:
            if stop_event and stop_event.is_set():
                break
            now = time.time()
            if (now - last_yielded) < min_interval:
                continue
            last_yielded = now
            arr = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            timestamp_ms = int((now - start_time) * 1000)
            yield frame, frame_index, timestamp_ms
            frame_index += 1


# ──────────────────────────────────────────────────────────────────────────────
# Frame persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_frames(
    source: Iterator[tuple[np.ndarray, int, int]],
    output_dir: Path | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[Path, list[dict]]:
    """
    Consume a frame iterator, write PNGs to disk, and return
    (frames_dir, manifest_list).

    manifest_list entries: {"index": int, "timestamp_ms": int, "path": str}
    """
    frames_dir = (output_dir or config.OUTPUT_DIR) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    for frame, index, ts in source:
        fname = frames_dir / f"frame_{index:06d}.png"
        cv2.imwrite(str(fname), frame)
        entry = {"index": index, "timestamp_ms": ts, "path": str(fname)}
        manifest.append(entry)
        if progress_callback:
            progress_callback(index)

    manifest_path = frames_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Saved %d frames → %s", len(manifest), frames_dir)
    return frames_dir, manifest
