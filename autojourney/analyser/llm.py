"""
LLM screen analyser — sends screenshots to a vision-capable LLM and
returns structured metadata about each screen.

Supports:
  - LM Studio local server (default, OpenAI-compatible API)
  - OpenAI API (set LLM_PROVIDER=openai)
  - Anthropic API (set LLM_PROVIDER=anthropic, uses openai-compat shim)

The analyser uses the same openai Python client for all providers by
pointing base_url at the appropriate endpoint.
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from openai import OpenAI

from autojourney import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert mobile UX analyst. You will be shown screenshots
from an iOS application. For each screenshot, return a JSON object with these exact keys:

{
  "app_name": "string — name of the app (from icon, nav bar, branding, or 'Unknown')",
  "screen_name": "string — name of this screen/page (from nav bar title, tab, or heading)",
  "ui_elements": ["list", "of", "visible", "interactive", "elements"],
  "inferred_action": "string — what the user probably did to arrive at this screen",
  "probable_destinations": ["list", "of", "likely", "next", "screens or actions"]
}

Return ONLY valid JSON, no commentary."""


def _build_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) based on LLM_PROVIDER config."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "openai":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        model = "gpt-4o"
    elif provider == "anthropic":
        # Anthropic has an OpenAI-compat endpoint
        client = OpenAI(
            api_key=config.ANTHROPIC_API_KEY or "dummy",
            base_url="https://api.anthropic.com/v1",
        )
        model = "claude-opus-4-5"
    else:
        # Default: LM Studio local server
        client = OpenAI(
            api_key="lm-studio",  # LM Studio ignores the key value
            base_url=config.LM_STUDIO_BASE_URL,
        )
        model = config.LM_STUDIO_MODEL

    return client, model


def _encode_image(image_path: Path) -> str:
    """Return base64-encoded PNG data URL."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def analyse_screen(image_path: Path) -> dict:
    """
    Send a single screenshot to the LLM and return the parsed JSON dict.

    Returns an empty dict with error key on failure.
    """
    client, model = _build_client()
    image_url = _encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": "Analyse this iOS screenshot and return the JSON.",
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            raw = raw.removeprefix("json")
        return json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        log.warning("LLM returned non-JSON for %s: %s", image_path.name, exc)
        return {"error": "json_decode", "raw": raw}
    except Exception as exc:  # noqa: BLE001 — deliberately broad: any provider/network failure degrades to an error dict rather than aborting the pipeline
        log.error("LLM request failed for %s: %s", image_path.name, exc)
        return {"error": str(exc)}
