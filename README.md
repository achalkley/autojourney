# AutoJourney

Capture a user's navigation through a 3rd-party iOS app and produce a visual journey map in Figma — including stitched scroll views, labelled screens, and inferred interaction flows.

## How it works

```
iOS Device (USB)
      ↓
[Capture Agent]     — streams H.264 from device via QuickTime USB protocol
      ↓                 (or reads a pre-recorded .mp4 file)
[Event Detector]    — detects screen transitions, modals, and scroll sequences
      ↓                 using SSIM, frame diff, and optical flow
[Scroll Stitcher]   — composites scroll sequences into full-height images
      ↓                 using template-matched pixel-row alignment
[LLM Analyser]      — sends each screen to a local vision model (LM Studio)
      ↓                 extracting app name, screen name, UI elements, actions
[Flow Graph]        — builds a directed tree of screens and transitions
      ↓
[Figma MCP Server]  — publishes the tree map to a Figma file via MCP
      ↓
[Markdown Report]   — writes journey-report.md with timestamped event log
```

## Requirements

- Python 3.11+
- macOS (for iOS USB capture)
- [LM Studio](https://lmstudio.ai/) with a vision-capable model loaded (e.g. LLaVA-Next)
- A Figma account — publishing uses [Figma's remote MCP server](https://developers.figma.com/docs/figma-mcp-server/) (`https://mcp.figma.com/mcp`); nothing to install or run locally, just a one-time browser sign-in on first publish
- For live USB capture: `pip install 'autojourney[capture]'`

## Setup

```bash
# Install
pip install -e .

# Copy and fill in your configuration
cp .env.example .env
```

Edit `.env`:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=llava-llama-3-v-vision   # model name as shown in LM Studio

FIGMA_FILE_KEY=your_file_key             # from the URL: figma.com/design/<FILE_KEY>/
```

No Figma token or local server to set up — the first `autojourney run` (or `publish`)
opens a browser to authorize AutoJourney against your Figma account via OAuth. The
resulting token is cached to `~/.config/autojourney/figma_oauth.json` and reused
(and refreshed) on later runs.

## Usage

### Process a recorded video

```bash
autojourney run --source recording.mp4
```

### Capture live from USB-connected iPhone

Connect your iPhone via USB, trust the computer, then:

```bash
autojourney run --usb
```

Press `Ctrl+C` when done navigating.

### Skip Figma publishing (local output only)

```bash
autojourney run --source recording.mp4 --no-publish
```

### Re-publish an existing session to Figma

```bash
autojourney publish output/session.json
```

### Re-generate the markdown report

```bash
autojourney report output/session.json
```

## Output files

After a run, `./output/` contains:

| File | Description |
|---|---|
| `frames/` | Extracted PNG frames |
| `frames/manifest.json` | Frame index with timestamps |
| `events.json` | Detected events (transitions, scrolls, etc.) |
| `screens/` | Per-screen PNGs (including stitched scroll images) |
| `session.json` | Full session data (screens, edges, LLM analysis) |
| `journey-report.md` | Human-readable timestamped event log |

## Configuration reference

See `.env.example` for all tunable parameters including:
- `TRANSITION_SSIM_THRESHOLD` — sensitivity for detecting screen transitions
- `SCROLL_FLOW_THRESHOLD` — optical flow magnitude to classify as scrolling
- `LLM_PROVIDER` — override to `openai` or `anthropic` for cloud LLMs

## Development

```bash
pip install -e '.[dev]'
pytest
```
