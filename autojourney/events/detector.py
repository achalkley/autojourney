"""
Event detection — analyses a sequence of frames and emits FrameEvents.

Detection strategy:
  - Screen transition: SSIM drops below threshold AND changed area fraction
    exceeds minimum. Indicates a full page/screen change.
  - Modal/overlay: Partial SSIM drop concentrated in a region (upper half
    stays stable, lower half or centre changes significantly).
  - Content update: Moderate partial change that doesn't meet transition
    criteria (e.g. data refresh, badge update).
  - Scroll: Optical flow shows dominant vertical or horizontal translation
    with low divergence (the content is moving uniformly in one direction).
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

from autojourney import config
from autojourney.models import EventType, FrameEvent

log = logging.getLogger(__name__)

# dominant_flow (dense Farneback optical flow) is by far the most expensive
# per-pair check. Its result is a median across the *whole* downsized frame,
# so if fewer than this fraction of pixels changed at all, the unchanged
# majority mathematically pulls that median toward zero regardless of what's
# happening in the changed region — it cannot cross any positive scroll_flow
# threshold. Below this floor, flow is skipped and treated as zero instead of
# computed. Guarded to only apply when scroll_flow is a real positive
# threshold, since the same reasoning doesn't hold at scroll_flow == 0 (any
# nonzero flow — including sub-pixel measurement noise in an otherwise-static
# region — would count).
SCROLL_FLOW_SKIP_AREA_FLOOR = 0.4


# ──────────────────────────────────────────────────────────────────────────────
# SSIM helper (single-channel)
# ──────────────────────────────────────────────────────────────────────────────

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity on grayscale images (manual Gaussian-weighted formula).

    `cv2.quality.QualitySSIM_compute` would be faster, but that module ships
    only in opencv-contrib-python, not the declared opencv-python dependency.
    """
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64) if a.ndim == 3 else a.astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64) if b.ndim == 3 else b.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.float32), (a.shape[1], a.shape[0])).astype(np.float64)

    C1, C2 = 6.5025, 58.5225  # (k1*255)^2, (k2*255)^2
    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(np.mean(num / den))


# ──────────────────────────────────────────────────────────────────────────────
# Changed-area fraction
# ──────────────────────────────────────────────────────────────────────────────

def changed_area_fraction(a: np.ndarray, b: np.ndarray, threshold: int = 30) -> float:
    """Fraction of pixels that differ by more than `threshold` in any channel."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    diff = cv2.absdiff(a, b)
    mask = np.any(diff > threshold, axis=2)
    return float(mask.sum()) / mask.size


# ──────────────────────────────────────────────────────────────────────────────
# Optical flow — scroll detection
# ──────────────────────────────────────────────────────────────────────────────

def dominant_flow(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Returns (mean_dx, mean_dy) of dense optical flow between frames.
    A dominant vertical flow with low divergence suggests a scroll.
    """
    g1 = cv2.cvtColor(cv2.resize(a, (320, 568)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(b, (320, 568)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))


# ──────────────────────────────────────────────────────────────────────────────
# Modal heuristic — stable top + changing centre/bottom
# ──────────────────────────────────────────────────────────────────────────────

def is_modal(a: np.ndarray, b: np.ndarray) -> bool:
    """True if the top quarter is stable but the lower 60% changed significantly."""
    h = a.shape[0]
    top_quarter = slice(0, h // 4)
    lower = slice(h // 4, h)
    top_ssim = ssim(a[top_quarter], b[top_quarter])
    lower_change = changed_area_fraction(a[lower], b[lower])
    return top_ssim > 0.90 and lower_change > 0.20


# ──────────────────────────────────────────────────────────────────────────────
# Main event detector
# ──────────────────────────────────────────────────────────────────────────────

class EventDetector:
    """
    Iterates over a manifest of frame paths and yields FrameEvents.

    Usage:
        detector = EventDetector(manifest, frames_dir)
        for event in detector.detect():
            process(event)
    """

    def __init__(
        self,
        manifest: list[dict],
        frames_dir: Path,
        transition_ssim: float | None = None,
        transition_area: float | None = None,
        scroll_flow: float | None = None,
    ) -> None:
        self.manifest = manifest
        self.frames_dir = frames_dir
        self.transition_ssim = (
            transition_ssim if transition_ssim is not None else config.TRANSITION_SSIM_THRESHOLD
        )
        self.transition_area = (
            transition_area if transition_area is not None else config.TRANSITION_AREA_FRACTION
        )
        self.scroll_flow = scroll_flow if scroll_flow is not None else config.SCROLL_FLOW_THRESHOLD

    def _load(self, entry: dict) -> np.ndarray | None:
        path = Path(entry["path"])
        frame = cv2.imread(str(path))
        if frame is None:
            log.warning("Cannot read frame, skipping: %s", path)
        return frame

    def detect(self) -> Generator[FrameEvent, None, None]:
        if not self.manifest:
            return

        scroll_accumulator: list[dict] = []
        in_scroll = False

        # An unreadable frame is treated as if it were never captured: it is
        # skipped entirely rather than compared against anything, so a single
        # corrupt frame doesn't abort the whole run.
        remaining = iter(self.manifest)
        prev_entry: dict | None = None
        prev_frame: np.ndarray | None = None
        for entry in remaining:
            frame = self._load(entry)
            if frame is not None:
                prev_entry, prev_frame = entry, frame
                break

        if prev_frame is None or prev_entry is None:
            yield FrameEvent(
                event_type=EventType.VIDEO_END,
                timestamp_ms=self.manifest[-1]["timestamp_ms"],
                frame_index=self.manifest[-1]["index"],
            )
            return

        for entry in remaining:
            curr_frame = self._load(entry)
            if curr_frame is None:
                continue
            ts = entry["timestamp_ms"]
            idx = entry["index"]

            s = ssim(prev_frame, curr_frame)
            area = changed_area_fraction(prev_frame, curr_frame)
            if self.scroll_flow > 0 and area < SCROLL_FLOW_SKIP_AREA_FLOOR:
                dx, dy = 0.0, 0.0
            else:
                dx, dy = dominant_flow(prev_frame, curr_frame)
            flow_mag = (dx ** 2 + dy ** 2) ** 0.5

            is_scroll_motion = flow_mag > self.scroll_flow and (abs(dy) > abs(dx) * 1.5 or abs(dx) > abs(dy) * 1.5)
            is_transition_change = s < self.transition_ssim and area > self.transition_area

            if is_scroll_motion:
                if not in_scroll:
                    in_scroll = True
                    scroll_accumulator = [prev_entry]
                    yield FrameEvent(
                        event_type=EventType.SCROLL_START,
                        timestamp_ms=ts,
                        frame_index=idx,
                        before_frame_path=Path(prev_entry["path"]),
                        metadata={"dx": dx, "dy": dy},
                    )
                scroll_accumulator.append(entry)
            else:
                if in_scroll:
                    in_scroll = False
                    yield FrameEvent(
                        event_type=EventType.SCROLL_END,
                        timestamp_ms=ts,
                        frame_index=idx,
                        scroll_frame_paths=[Path(e["path"]) for e in scroll_accumulator],
                        metadata={"frame_count": len(scroll_accumulator)},
                    )
                    scroll_accumulator = []

                if is_transition_change:
                    if is_modal(prev_frame, curr_frame):
                        evt_type = EventType.MODAL
                    else:
                        evt_type = EventType.TRANSITION
                    yield FrameEvent(
                        event_type=evt_type,
                        timestamp_ms=ts,
                        frame_index=idx,
                        before_frame_path=Path(prev_entry["path"]),
                        after_frame_path=Path(entry["path"]),
                        metadata={"ssim": s, "changed_area": area},
                    )
                elif area > 0.05:
                    yield FrameEvent(
                        event_type=EventType.CONTENT_UPDATE,
                        timestamp_ms=ts,
                        frame_index=idx,
                        before_frame_path=Path(prev_entry["path"]),
                        after_frame_path=Path(entry["path"]),
                        metadata={"ssim": s, "changed_area": area},
                    )

            prev_entry = entry
            prev_frame = curr_frame

        # Close any open scroll at end of video
        if in_scroll and scroll_accumulator:
            yield FrameEvent(
                event_type=EventType.SCROLL_END,
                timestamp_ms=self.manifest[-1]["timestamp_ms"],
                frame_index=self.manifest[-1]["index"],
                scroll_frame_paths=[Path(e["path"]) for e in scroll_accumulator],
                metadata={"frame_count": len(scroll_accumulator)},
            )

        yield FrameEvent(
            event_type=EventType.VIDEO_END,
            timestamp_ms=self.manifest[-1]["timestamp_ms"],
            frame_index=self.manifest[-1]["index"],
        )
