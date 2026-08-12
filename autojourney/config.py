"""
Shared configuration loaded from environment / .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


def _int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


# LLM
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "lmstudio")
LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL: str = os.getenv("LM_STUDIO_MODEL", "llava-llama-3-v-vision")
LM_STUDIO_API_KEY: str | None = os.getenv("LM_STUDIO_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# Figma
# The remote MCP server (auth is OAuth, done once via browser — no token to configure)
FIGMA_MCP_SERVER_URL: str = os.getenv("FIGMA_MCP_SERVER_URL", "https://mcp.figma.com/mcp")
FIGMA_FILE_KEY: str = os.getenv("FIGMA_FILE_KEY", "")
FIGMA_PAGE_NAME: str = os.getenv("FIGMA_PAGE_NAME", "AutoJourney")
FIGMA_OAUTH_TOKEN_CACHE: Path = Path(
    os.getenv("FIGMA_OAUTH_TOKEN_CACHE", str(Path.home() / ".config" / "autojourney" / "figma_oauth.json"))
)

# Output
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output"))

# Event detection
TRANSITION_SSIM_THRESHOLD: float = _float("TRANSITION_SSIM_THRESHOLD", 0.70)
TRANSITION_AREA_FRACTION: float = _float("TRANSITION_AREA_FRACTION", 0.15)
SCROLL_FLOW_THRESHOLD: float = _float("SCROLL_FLOW_THRESHOLD", 3.0)
