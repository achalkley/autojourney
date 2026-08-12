"""
Tests for the CLI entry point.
"""
from __future__ import annotations

from click.testing import CliRunner

from autojourney.cli.main import cli


# ──────────────────────────────────────────────────────────────────────────────
# Regression coverage for known defects (audit Phase 2: fps validation)
# ──────────────────────────────────────────────────────────────────────────────

class TestFpsValidation:
    """
    --fps 0 used to reach frames_from_file / frames_from_usb, both of which
    divide by fps_limit, raising ZeroDivisionError deep inside a video read.
    It should be rejected at the CLI boundary instead, before any capture work
    starts.
    """

    def test_zero_fps_rejected_before_capture_starts(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--usb", "--fps", "0"])
        assert result.exit_code != 0
        assert "ZeroDivisionError" not in (result.output + str(result.exception))
        assert "fps" in result.output.lower()

    def test_negative_fps_rejected(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--usb", "--fps", "-1"])
        assert result.exit_code != 0
        assert "fps" in result.output.lower()

    def test_positive_fps_accepted_by_option_parsing(self):
        """
        A valid --fps shouldn't be rejected by option parsing. (The command
        will still fail past that point in this test — there's no real USB
        device — but that failure must not be the fps validation.)
        """
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--usb", "--fps", "5"])
        assert "Invalid value for '--fps'" not in result.output
