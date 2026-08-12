"""
Markdown report generator — produces a human-readable journey-report.md
from a completed JourneySession.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from autojourney.models import JourneySession

log = logging.getLogger(__name__)


def _ms_to_time(ms: int) -> str:
    """Format milliseconds as MM:SS.mmm"""
    total_s, millis = divmod(ms, 1000)
    minutes, seconds = divmod(total_s, 60)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _escape_table_cell(value: str) -> str:
    """
    Neutralise markdown table syntax in cell content.

    Screen metadata comes from an LLM response and isn't sanitised upstream —
    a literal `|` splits the row into extra columns, and a literal newline
    terminates it early, both corrupting every row that follows.
    """
    return value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def generate_report(session: JourneySession, output_path: Path) -> None:
    """
    Write a markdown event log to output_path.

    Columns: timestamp | screen | app | inferred action | UI elements
    """
    edge_by_dest = {e.to_screen_id: e for e in session.edges}

    lines: list[str] = [
        "# AutoJourney Session Report",
        "",
        f"**Session ID:** `{session.session_id}`  ",
        f"**Source:** `{session.source_path}`  ",
        f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Screens captured:** {len(session.screens)}  ",
        f"**Transitions:** {len(session.edges)}  ",
        "",
        "---",
        "",
        "## Journey Log",
        "",
        "| Time | Screen | App | Action taken | Key UI elements |",
        "|---|---|---|---|---|",
    ]

    for screen in session.screens:
        ts = _ms_to_time(screen.timestamp_ms)
        screen_name = screen.screen_name or f"*(unlabelled — {screen.screen_id})*"
        app = screen.app_name or "—"
        # Edge incoming to this screen describes the action that caused it
        edge = edge_by_dest.get(screen.screen_id)
        action = edge.interaction_label if edge else screen.inferred_action or "*(session start)*"
        elements = ", ".join(screen.ui_elements[:5]) if screen.ui_elements else "—"
        cells = (_escape_table_cell(c) for c in (ts, screen_name, app, action, elements))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "---",
        "",
        "## Screen Detail",
        "",
    ]

    for screen in session.screens:
        lines += [
            f"### {screen.screen_name or screen.screen_id}",
            "",
            f"- **App:** {screen.app_name or '—'}",
            f"- **Time:** {_ms_to_time(screen.timestamp_ms)}",
            f"- **Event type:** `{screen.event_type}`",
            f"- **Inferred action:** {screen.inferred_action or '—'}",
            f"- **UI elements:** {', '.join(screen.ui_elements) if screen.ui_elements else '—'}",
            f"- **Probable next destinations:** {', '.join(screen.probable_destinations) if screen.probable_destinations else '—'}",
            f"- **Image:** `{screen.image_path.name}`",
            "",
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written → %s", output_path)
