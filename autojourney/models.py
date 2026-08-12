"""
Shared data models used across pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    TRANSITION = "transition"    # Full screen change
    MODAL = "modal"              # Partial overlay appeared/disappeared
    CONTENT_UPDATE = "content"   # Content changed without full transition
    SCROLL_START = "scroll_start"
    SCROLL_END = "scroll_end"
    VIDEO_END = "video_end"


@dataclass
class FrameEvent:
    """A detected event in the video stream."""
    event_type: EventType
    timestamp_ms: int            # Milliseconds from video start
    frame_index: int
    before_frame_path: Path | None = None
    after_frame_path: Path | None = None
    # For scroll events — list of frame paths spanning the scroll
    scroll_frame_paths: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Screen:
    """A single logical screen, possibly a stitched scroll image."""
    screen_id: str
    image_path: Path
    timestamp_ms: int
    event_type: EventType
    # Populated by LLM analyser
    app_name: str = ""
    screen_name: str = ""
    ui_elements: list[str] = field(default_factory=list)
    inferred_action: str = ""       # What the user probably did to arrive here
    probable_destinations: list[str] = field(default_factory=list)
    raw_llm_response: str = ""


@dataclass
class FlowEdge:
    """A directed transition between two screens."""
    from_screen_id: str
    to_screen_id: str
    interaction_label: str
    timestamp_ms: int


@dataclass
class JourneySession:
    """Complete processed journey session."""
    session_id: str
    source_path: Path            # Video file or 'usb_stream'
    screens: list[Screen] = field(default_factory=list)
    edges: list[FlowEdge] = field(default_factory=list)
