"""
Tests for LM Studio / cloud-provider client construction in analyser/llm.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from autojourney import config
from autojourney.analyser import llm as llm_module
from autojourney.analyser.llm import analyse_screen

_FAKE_RESPONSE_JSON = json.dumps({
    "app_name": "TestApp",
    "screen_name": "Home",
    "ui_elements": [],
    "inferred_action": "opened app",
    "probable_destinations": [],
})


class _RecordingOpenAI:
    """Stand-in for openai.OpenAI: records constructor kwargs and returns a
    canned chat completion, so analyse_screen's full path (client
    construction -> request -> JSON parse) runs with no network call."""

    instances: ClassVar[list[_RecordingOpenAI]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        message = type("Message", (), {"content": _FAKE_RESPONSE_JSON})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


@pytest.fixture
def recording_openai(monkeypatch):
    _RecordingOpenAI.instances.clear()
    monkeypatch.setattr(llm_module, "OpenAI", _RecordingOpenAI)
    return _RecordingOpenAI


@pytest.fixture
def fake_screenshot(tmp_path) -> Path:
    path = tmp_path / "screen.png"
    path.write_bytes(b"not-a-real-png-but-encode-doesnt-care")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# LM Studio optional bearer-token auth
# ──────────────────────────────────────────────────────────────────────────────

def test_lm_studio_falls_back_to_placeholder_when_key_unset(recording_openai, fake_screenshot, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "lmstudio")
    monkeypatch.setattr(config, "LM_STUDIO_API_KEY", None)
    monkeypatch.setattr(config, "LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

    result = analyse_screen(fake_screenshot)

    kwargs = recording_openai.instances[-1].kwargs
    assert kwargs["api_key"] == "lm-studio"
    assert kwargs["base_url"] == "http://localhost:1234/v1"
    assert result["app_name"] == "TestApp"  # proves the client was actually used


def test_lm_studio_uses_configured_api_key_when_set(recording_openai, fake_screenshot, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "lmstudio")
    monkeypatch.setattr(config, "LM_STUDIO_API_KEY", "sk-local-abc123")

    analyse_screen(fake_screenshot)

    assert recording_openai.instances[-1].kwargs["api_key"] == "sk-local-abc123"


def test_lm_studio_api_key_does_not_leak_into_openai_provider(recording_openai, fake_screenshot, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LM_STUDIO_API_KEY", "sk-local-should-not-appear")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-openai-real")

    analyse_screen(fake_screenshot)

    assert recording_openai.instances[-1].kwargs["api_key"] == "sk-openai-real"
