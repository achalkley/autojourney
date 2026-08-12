"""
Tests for markdown report generator.
"""
from __future__ import annotations

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
