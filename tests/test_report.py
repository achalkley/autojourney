"""
Tests for markdown report generator.
"""
from __future__ import annotations

import re
from pathlib import Path

from autojourney.models import EventType, FlowEdge, JourneySession, Screen
from autojourney.report.markdown import generate_report


def _make_session() -> JourneySession:
    screens = [
        Screen(
            "s0001", Path("a.png"), 0, EventType.TRANSITION,
            app_name="ShopApp", screen_name="Home",
            ui_elements=["Search bar", "Featured items"],
            inferred_action="App opened",
            probable_destinations=["Search", "Product detail"],
        ),
        Screen(
            "s0002", Path("b.png"), 1500, EventType.TRANSITION,
            app_name="ShopApp", screen_name="Product Detail",
            ui_elements=["Add to cart", "Images", "Description"],
            inferred_action="Tapped product",
            probable_destinations=["Cart", "Back"],
        ),
    ]
    edges = [FlowEdge("s0001", "s0002", "Tapped product", 1500)]
    return JourneySession("abc123", Path("recording.mp4"), screens=screens, edges=edges)


class TestGenerateReport:
    def test_report_file_created(self, tmp_path):
        session = _make_session()
        out = tmp_path / "report.md"
        generate_report(session, out)
        assert out.exists()

    def test_report_contains_screen_names(self, tmp_path):
        session = _make_session()
        out = tmp_path / "report.md"
        generate_report(session, out)
        content = out.read_text()
        assert "Home" in content
        assert "Product Detail" in content
        assert "ShopApp" in content

    def test_report_contains_interaction_label(self, tmp_path):
        session = _make_session()
        out = tmp_path / "report.md"
        generate_report(session, out)
        content = out.read_text()
        assert "Tapped product" in content

    def test_report_contains_session_id(self, tmp_path):
        session = _make_session()
        out = tmp_path / "report.md"
        generate_report(session, out)
        content = out.read_text()
        assert "abc123" in content

    def test_report_is_valid_markdown(self, tmp_path):
        session = _make_session()
        out = tmp_path / "report.md"
        generate_report(session, out)
        content = out.read_text()
        # Basic markdown checks
        assert content.startswith("# ")
        assert "| Time | Screen |" in content


# ──────────────────────────────────────────────────────────────────────────────
# Regression coverage for known defects (audit Phase 2: table cell escaping)
# ──────────────────────────────────────────────────────────────────────────────

class TestTableCellEscaping:
    """
    Screen metadata is unsanitised LLM output. A literal `|` splits a row into
    extra columns; a literal newline terminates it early and starts a new one.
    Either corrupts the table for every row after it — assert through the
    public entry point on the actual rendered table, not the escaping helper.
    """

    def _journey_log_rows(self, content: str) -> list[str]:
        lines = content.splitlines()
        header = "| Time | Screen | App | Action taken | Key UI elements |"
        start = lines.index(header) + 2  # skip header + '|---|...' separator
        end = next(i for i in range(start, len(lines)) if not lines[i].startswith("|"))
        return lines[start:end]

    def test_pipe_in_llm_text_does_not_split_columns(self, tmp_path):
        session = _make_session()
        session.screens[0].app_name = "Shop | App"

        out = tmp_path / "report.md"
        generate_report(session, out)
        content = out.read_text()

        rows = self._journey_log_rows(content)
        assert len(rows) == len(session.screens), (
            f"expected {len(session.screens)} table rows, got {len(rows)}: {rows!r}"
        )
        # Exactly 6 unescaped pipes per row (5 columns) — a literal "|" that
        # leaked through unescaped would push this to 7+.
        for row in rows:
            unescaped_pipes = re.findall(r"(?<!\\)\|", row)
            assert len(unescaped_pipes) == 6, (
                f"row has {len(unescaped_pipes)} unescaped '|' (expected 6): {row!r}"
            )
        assert "Shop \\| App" in content

    def test_newline_in_llm_text_does_not_add_a_row(self, tmp_path):
        session = _make_session()
        session.screens[0].inferred_action = "Tapped\nbutton"

        out = tmp_path / "report.md"
        generate_report(session, out)
        content = out.read_text()

        rows = self._journey_log_rows(content)
        assert len(rows) == len(session.screens), (
            f"expected {len(session.screens)} table rows, got {len(rows)}: {rows!r}"
        )
        assert "Tapped button" in content
