"""
AutoJourney CLI entry point.

Usage:
    autojourney run --source session.mp4
    autojourney run --usb
    autojourney run --source session.mp4 --no-publish
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )
    # Silence overly chatty third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@click.group()
def cli() -> None:
    """AutoJourney — capture iOS app sessions and generate Figma journey maps."""
    pass


@cli.command()
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a pre-recorded .mp4 / .mov video file.",
)
@click.option(
    "--usb",
    is_flag=True,
    default=False,
    help="Capture live from a USB-connected iOS device.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: ./output).",
)
@click.option(
    "--fps",
    type=click.FloatRange(min=0, min_open=True),
    default=5.0,
    show_default=True,
    help="Frames per second to sample from the video.",
)
@click.option(
    "--no-publish",
    is_flag=True,
    default=False,
    help="Skip Figma publishing; only produce local output files.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def run(
    source: Path | None,
    usb: bool,
    output: Path | None,
    fps: float,
    no_publish: bool,
    verbose: bool,
) -> None:
    """Run the full AutoJourney pipeline."""
    _setup_logging(verbose)

    if not source and not usb:
        console.print("[red]Error:[/red] Provide --source <video.mp4> or --usb")
        sys.exit(1)
    if source and usb:
        console.print("[red]Error:[/red] --source and --usb are mutually exclusive")
        sys.exit(1)

    from autojourney.pipeline import run_pipeline
    from autojourney import config

    out_dir = output or config.OUTPUT_DIR

    console.print(Panel.fit(
        f"[bold]AutoJourney[/bold]\n"
        f"Source: {'USB live capture' if usb else str(source)}\n"
        f"Output: {out_dir}\n"
        f"Figma publish: {'no' if no_publish else 'yes'}",
        title="🚀 Starting",
    ))

    def on_progress(stage: str, detail: str) -> None:
        icon = {
            "capture": "📹",
            "events": "🔍",
            "stitch": "🧵",
            "analyse": "🤖",
            "graph": "🗺️",
            "figma": "🎨",
            "report": "📄",
        }.get(stage, "•")
        console.print(f"  {icon} [bold]{stage}[/bold]: {detail}")

    try:
        session = run_pipeline(
            video_path=source,
            output_dir=out_dir,
            publish=not no_publish,
            fps_limit=fps,
            on_progress=on_progress,
        )
        console.print(Panel.fit(
            f"[green]✓ Done[/green]\n"
            f"Screens: {len(session.screens)}\n"
            f"Transitions: {len(session.edges)}\n"
            f"Report: {out_dir / 'journey-report.md'}\n"
            f"Session data: {out_dir / 'session.json'}",
            title="✅ Complete",
        ))
    except Exception as exc:
        console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("session_json", type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, default=False)
def publish(session_json: Path, verbose: bool) -> None:
    """Re-publish an existing session.json to Figma."""
    _setup_logging(verbose)

    import json
    from autojourney.models import FlowEdge, JourneySession, Screen
    from autojourney.flow.graph import build_graph, compute_tree_layout
    from autojourney.figma.publisher import publish_to_figma

    raw = json.loads(session_json.read_text())
    session = JourneySession(
        session_id=raw["session_id"],
        source_path=Path(raw["source"]),
        screens=[
            Screen(
                screen_id=s["id"],
                image_path=Path(s["image"]),
                timestamp_ms=s["timestamp_ms"],
                event_type=s["event_type"],
                app_name=s.get("app_name", ""),
                screen_name=s.get("screen_name", ""),
                ui_elements=s.get("ui_elements", []),
                inferred_action=s.get("inferred_action", ""),
                probable_destinations=s.get("probable_destinations", []),
            )
            for s in raw.get("screens", [])
        ],
        edges=[
            FlowEdge(
                from_screen_id=e["from"],
                to_screen_id=e["to"],
                interaction_label=e["label"],
                timestamp_ms=e["timestamp_ms"],
            )
            for e in raw.get("edges", [])
        ],
    )

    graph = build_graph(session)
    layout = compute_tree_layout(graph)
    url = publish_to_figma(session, layout)
    console.print(f"[green]Published:[/green] {url}")


@cli.command()
@click.argument("session_json", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for the markdown report.",
)
def report(session_json: Path, output: Path | None) -> None:
    """(Re-)generate the markdown report from an existing session.json."""
    import json
    from autojourney.models import FlowEdge, JourneySession, Screen
    from autojourney.report.markdown import generate_report

    raw = json.loads(session_json.read_text())
    session = JourneySession(
        session_id=raw["session_id"],
        source_path=Path(raw["source"]),
        screens=[
            Screen(
                screen_id=s["id"],
                image_path=Path(s["image"]),
                timestamp_ms=s["timestamp_ms"],
                event_type=s["event_type"],
                app_name=s.get("app_name", ""),
                screen_name=s.get("screen_name", ""),
                ui_elements=s.get("ui_elements", []),
                inferred_action=s.get("inferred_action", ""),
                probable_destinations=s.get("probable_destinations", []),
            )
            for s in raw.get("screens", [])
        ],
        edges=[
            FlowEdge(
                from_screen_id=e["from"],
                to_screen_id=e["to"],
                interaction_label=e["label"],
                timestamp_ms=e["timestamp_ms"],
            )
            for e in raw.get("edges", [])
        ],
    )
    out = output or (session_json.parent / "journey-report.md")
    generate_report(session, out)
    console.print(f"[green]Report written:[/green] {out}")
