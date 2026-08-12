"""
Scroll stitcher — combines a sequence of frames captured during a scroll
into a single tall composite image.

Algorithm:
  1. For each consecutive pair of frames in the scroll sequence:
     a. Use template matching to find the vertical (or horizontal) offset
        where the content from frame N aligns with frame N+1.
     b. Crop the non-overlapping strip from frame N+1 and append it.
  2. Return the composited tall image as a numpy array.

Works best with:
  - Vertical scrolls (most common on iOS)
  - Horizontal scrolls (carousels, tab swiping)
  - Minimal UI chrome changes during the scroll
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Fraction of the frame height/width to use as the matching template
TEMPLATE_FRACTION = 0.25
# Search range: how far (in pixels) to look for the match beyond the expected region
SEARCH_MARGIN = 80
# Number of leading consecutive frame pairs to sample for direction detection
DIRECTION_SAMPLE_PAIRS = 4


def _match(frame_a: np.ndarray, frame_b: np.ndarray, direction: str) -> tuple[float, int]:
    """
    Template-match frame_a's trailing edge (bottom row or right column) against
    frame_b's leading edge, in the given direction.

    Returns (confidence, unique_pixels):
      - confidence: the best normalized cross-correlation score (higher = more
        confident this is a true alignment, not a coincidental one).
      - unique_pixels: the number of non-overlapping rows/cols to take from
        frame_b to extend a stitch in this direction.
    """
    h, w = frame_a.shape[:2]

    if direction == "vertical":
        # Template = bottom TEMPLATE_FRACTION of frame_a (greyscale)
        tmpl_h = int(h * TEMPLATE_FRACTION)
        template = cv2.cvtColor(frame_a[h - tmpl_h:, :], cv2.COLOR_BGR2GRAY)
        # Search space = top portion of frame_b
        search_h = tmpl_h + SEARCH_MARGIN
        search = cv2.cvtColor(frame_b[:min(search_h + tmpl_h, h), :], cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        match_top = max_loc[1]  # row in search where template starts
        # Unique rows from frame_b = rows below where the overlap ends
        overlap_end_in_b = match_top + tmpl_h
        return float(max_val), max(1, h - overlap_end_in_b)
    else:
        tmpl_w = int(w * TEMPLATE_FRACTION)
        template = cv2.cvtColor(frame_a[:, w - tmpl_w:], cv2.COLOR_BGR2GRAY)
        search_w = tmpl_w + SEARCH_MARGIN
        search = cv2.cvtColor(frame_b[:, :min(search_w + tmpl_w, w)], cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        match_left = max_loc[0]
        overlap_end_in_b = match_left + tmpl_w
        return float(max_val), max(1, w - overlap_end_in_b)


def _detect_scroll_direction(frames: list[np.ndarray]) -> str:
    """
    Determine dominant scroll direction by template-match confidence, voted
    across the first few consecutive frame pairs.

    Optical flow (the original approach) was dropped: Farneback's default
    parameters can't resolve the frame-to-frame displacement typical of a
    downsampled scroll capture, so on a long sequence its flow estimate is
    noise and whichever axis's noise happens to be larger "wins" arbitrarily
    — confirmed to misclassify a pure vertical scroll as horizontal once a
    sequence runs past ~25 frames. Template matching is the same primitive
    `_find_overlap_offset` already relies on for the stitch itself, and
    unlike Farneback its confidence score isn't sensitive to displacement
    size. Voting across several pairs (rather than just the first) guards
    against the first pair happening to have near-zero motion, which is
    likely right at a SCROLL_START boundary.
    """
    if len(frames) < 2:
        return "vertical"

    v_total = h_total = 0.0
    for a, b in list(zip(frames, frames[1:], strict=False))[:DIRECTION_SAMPLE_PAIRS]:
        v_total += _match(a, b, "vertical")[0]
        h_total += _match(a, b, "horizontal")[0]
    return "vertical" if v_total >= h_total else "horizontal"


def _find_overlap_offset(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    direction: str,
) -> int:
    """
    Return the pixel offset (row for vertical, col for horizontal) in frame_b
    where the bottom portion of frame_a best matches the top portion of frame_b.

    Returns a positive integer: the number of unique (non-overlapping) rows/cols
    to take from frame_b to extend the stitch.
    """
    _, unique_pixels = _match(frame_a, frame_b, direction)
    return unique_pixels


def stitch_scroll(
    frame_paths: list[Path],
    output_path: Path | None = None,
) -> np.ndarray:
    """
    Stitch a list of scroll frames into one tall (or wide) image.

    Args:
        frame_paths: Ordered list of frame image paths (scroll sequence).
        output_path: If provided, save the stitched PNG here.

    Returns:
        Composited BGR image as numpy array.
    """
    if not frame_paths:
        raise ValueError("No frames to stitch")

    frames = [cv2.imread(str(p)) for p in frame_paths]
    frames = [f for f in frames if f is not None]

    if len(frames) == 1:
        if output_path:
            cv2.imwrite(str(output_path), frames[0])
        return frames[0]

    direction = _detect_scroll_direction(frames)
    log.info("Stitching %d frames (%s scroll) …", len(frames), direction)

    # Start with the first frame
    composite = frames[0]

    for i, frame_b in enumerate(frames[1:], start=1):
        unique_pixels = _find_overlap_offset(frames[i - 1], frame_b, direction)

        if direction == "vertical":
            h = frame_b.shape[0]
            strip = frame_b[h - unique_pixels:, :]
            composite = np.vstack([composite, strip])
        else:
            w = frame_b.shape[1]
            strip = frame_b[:, w - unique_pixels:]
            composite = np.hstack([composite, strip])

    log.info("Stitched image size: %s", composite.shape)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), composite)
        log.info("Saved stitched image → %s", output_path)

    return composite
