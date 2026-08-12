"""
Pipeline orchestrator — wires all stages together.

Usage (from CLI or tests):
    from autojourney.pipeline import run_pipeline
    run_pipeline(video_path=Path("session.mp4"), output_dir=Path("output"))
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path

import cv2

from autojourney import config
from autojourney.analyser.llm import analyse_screen
from autojourney.capture.agent import frames_from_file, save_frames
from autojourney.events.detector import EventDetector
from autojourney.figma.publisher import publish_to_figma
from autojourney.flow.graph import build_graph, compute_tree_layout
from autojourney.models import (
    EventType,
    FlowEdge,
    FrameEvent,
    JourneySession,
    Screen,
)
from autojourney.report.markdown import generate_report
from autojourney.stitcher.scroll import stitch_scroll

log = logging.getLogger(__name__)


def _screen_id(index: int) -> str:
    return f"s{index:04d}"


def _copy_as_png(src: Path, dest: Path) -> None:
    """
    Re-encode src as PNG at dest, regardless of src's own format.

    Captured frames may be JPEG (see capture/agent.py), but screens are
    analysed by a vision model and optionally uploaded to Figma — both
    consumers assume the screens/ directory is PNG, so a raw byte copy
    here would silently mislabel JPEG bytes with a .png extension.
    """
    frame = cv2.imread(str(src))
    if frame is None:
        raise RuntimeError(f"Could not read frame for screen re-encode: {src}")
    cv2.imwrite(str(dest), frame)


def run_pipeline(
    video_path: Path | None = None,
    output_dir: Path | None = None,
    publish: bool = True,
    fps_limit: float = 5.0,
    on_progress: Callable[[str, str], None] | None = None,
) -> JourneySession:
    """
    Full pipeline:
      1. Extract frames from video (or USB stream)
      2. Detect events
      3. Stitch scroll sequences
      4. Analyse screens with LLM
      5. Build flow graph
      6. Publish to Figma
      7. Write markdown report

    Returns the completed JourneySession.
    """
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())[:8]
    source_path = video_path or Path("usb_stream")
    session = JourneySession(session_id=session_id, source_path=source_path)

    def progress(stage: str, detail: str = "") -> None:
        log.info("[%s] %s", stage, detail)
        if on_progress:
            on_progress(stage, detail)

    # ── Stage 1: Frame extraction ────────────────────────────────────────────
    progress("capture", f"Extracting frames from {source_path}")
    if video_path:
        frame_iter = frames_from_file(video_path, fps_limit=fps_limit)
    else:
        from autojourney.capture.agent import frames_from_usb
        frame_iter = frames_from_usb(fps_limit=fps_limit)

    # Ctrl+C during live USB capture is the documented way to end a capture
    # session (see README) — treat it as "stop capturing", not an abort, and
    # continue into the rest of the pipeline with whatever was captured.
    frames_dir, manifest = save_frames(frame_iter, output_dir=out, swallow_interrupt=video_path is None)
    progress("capture", f"Extracted {len(manifest)} frames")

    if not manifest:
        log.warning("No frames extracted — aborting")
        return session

    # ── Stage 2: Event detection ─────────────────────────────────────────────
    progress("events", "Detecting screen transitions and scroll events")
    detector = EventDetector(manifest, frames_dir)
    events: list[FrameEvent] = list(detector.detect())
    progress("events", f"Detected {len(events)} events")

    # Persist events
    events_path = out / "events.json"
    events_path.write_text(
        json.dumps(
            [
                {
                    "type": e.event_type,
                    "timestamp_ms": e.timestamp_ms,
                    "frame_index": e.frame_index,
                }
                for e in events
            ],
            indent=2,
        )
    )

    # ── Stage 3: Scroll stitching & screen collection ────────────────────────
    progress("stitch", "Stitching scroll sequences and collecting screens")
    screens_dir = out / "screens"
    screens_dir.mkdir(exist_ok=True)

    screen_index = 0
    prev_screen_id: str | None = None

    def _add_screen(image_path: Path, event: FrameEvent) -> Screen:
        nonlocal screen_index, prev_screen_id
        sid = _screen_id(screen_index)
        screen = Screen(
            screen_id=sid,
            image_path=image_path,
            timestamp_ms=event.timestamp_ms,
            event_type=event.event_type,
        )
        session.screens.append(screen)
        if prev_screen_id is not None:
            session.edges.append(
                FlowEdge(
                    from_screen_id=prev_screen_id,
                    to_screen_id=sid,
                    interaction_label=event.event_type.value,
                    timestamp_ms=event.timestamp_ms,
                )
            )
        prev_screen_id = sid
        screen_index += 1
        return screen

    # Seed the screen the journey opens on. TRANSITION/MODAL events only ever
    # add the *after*-frame, and SCROLL_END/VIDEO_END don't cover frame 0
    # either, so without this the starting screen — and the root node of the
    # journey graph — is dropped whenever any transition occurs.
    opening_entry = manifest[0]
    opening_dest = screens_dir / f"screen_{screen_index:04d}.png"
    _copy_as_png(Path(opening_entry["path"]), opening_dest)
    _add_screen(
        opening_dest,
        FrameEvent(
            event_type=EventType.TRANSITION,
            timestamp_ms=opening_entry["timestamp_ms"],
            frame_index=opening_entry["index"],
        ),
    )

    for event in events:
        if event.event_type == EventType.SCROLL_END and event.scroll_frame_paths:
            stitch_path = screens_dir / f"scroll_{screen_index:04d}.png"
            stitch_scroll(event.scroll_frame_paths, output_path=stitch_path)
            _add_screen(stitch_path, event)

        elif event.event_type in (EventType.TRANSITION, EventType.MODAL):
            if event.after_frame_path:
                dest = screens_dir / f"screen_{screen_index:04d}.png"
                _copy_as_png(event.after_frame_path, dest)
                _add_screen(dest, event)

    progress("stitch", f"Collected {len(session.screens)} screens")

    # ── Stage 4: LLM screen analysis ─────────────────────────────────────────
    progress("analyse", f"Analysing {len(session.screens)} screens with LLM")
    for i, screen in enumerate(session.screens):
        progress("analyse", f"Screen {i+1}/{len(session.screens)}: {screen.image_path.name}")
        result = analyse_screen(screen.image_path)
        if "error" not in result:
            screen.app_name = result.get("app_name", "")
            screen.screen_name = result.get("screen_name", "")
            screen.ui_elements = result.get("ui_elements", [])
            screen.inferred_action = result.get("inferred_action", "")
            screen.probable_destinations = result.get("probable_destinations", [])
        screen.raw_llm_response = json.dumps(result)

        # Backfill transition edge labels now that we know screen names
        if i > 0:
            edge = session.edges[i - 1] if i - 1 < len(session.edges) else None
            if (
                edge
                and (not edge.interaction_label or edge.interaction_label == EventType.TRANSITION.value)
                and screen.inferred_action
            ):
                edge.interaction_label = screen.inferred_action

    # Persist session JSON
    session_path = out / "session.json"
    session_path.write_text(
        json.dumps(
            {
                "session_id": session.session_id,
                "source": str(session.source_path),
                "screens": [
                    {
                        "id": s.screen_id,
                        "image": str(s.image_path),
                        "app_name": s.app_name,
                        "screen_name": s.screen_name,
                        "timestamp_ms": s.timestamp_ms,
                        "event_type": s.event_type,
                        "inferred_action": s.inferred_action,
                        "ui_elements": s.ui_elements,
                        "probable_destinations": s.probable_destinations,
                    }
                    for s in session.screens
                ],
                "edges": [
                    {
                        "from": e.from_screen_id,
                        "to": e.to_screen_id,
                        "label": e.interaction_label,
                        "timestamp_ms": e.timestamp_ms,
                    }
                    for e in session.edges
                ],
            },
            indent=2,
        )
    )
    progress("analyse", f"Session saved → {session_path}")

    # ── Stage 5: Flow graph ───────────────────────────────────────────────────
    progress("graph", "Building flow graph")
    graph = build_graph(session)
    layout = compute_tree_layout(graph)

    # ── Stage 6: Figma publish ────────────────────────────────────────────────
    if publish and config.FIGMA_FILE_KEY:
        progress("figma", "Publishing to Figma via the remote MCP server")
        try:
            figma_url = publish_to_figma(
                session=session,
                layout=layout,
                progress_callback=lambda done, total, msg: progress("figma", f"[{done}/{total}] {msg}"),
            )
            progress("figma", f"Published: {figma_url}")
        except Exception as exc:  # noqa: BLE001 — a publish failure shouldn't discard an otherwise-complete local session
            log.error("Figma publish failed: %s", exc)
            progress("figma", f"ERROR: {exc}")
    else:
        if publish:
            log.warning("Figma credentials not configured — skipping publish")

    # ── Stage 7: Markdown report ──────────────────────────────────────────────
    report_path = out / "journey-report.md"
    generate_report(session, report_path)
    progress("report", f"Report written → {report_path}")

    return session
