# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e '.[dev]'    # editable install + pytest, ruff, mypy
pytest                     # full suite
pytest tests/test_stitcher.py::TestStitchScroll::test_output_file_written   # single test
ruff check .
mypy autojourney
```

Without an editable install, `pytest` fails at collection — there is no `conftest.py` and the test files have no `__init__.py`, so pytest puts `tests/` on `sys.path` rather than the repo root and `import autojourney` fails. Use `python -m pytest` (which prepends the cwd) if you need to run against a bare checkout.

CI (`.github/workflows/ci.yml`) runs `ruff check .` and `pytest` as hard gates on push/PR to `main`. `mypy autojourney` also runs but is advisory (`continue-on-error`) — the tree has pre-existing `cv2`-stub-shaped errors (`calcOpticalFlowFarneback` overload mismatches, `Optional` propagation through `stitcher/scroll.py` from `cv2.imread`'s `None` return) that aren't fixed yet. Deliberate `ruff` exceptions (a handful of intentionally broad `except Exception` fallbacks, one dependent triple `async with`) are silenced with scoped inline `# noqa: <CODE>` comments, not a blanket per-rule ignore, so the rule stays live everywhere else.

The `capture` extra (`pip install -e '.[capture]'`) is only needed for live USB capture; everything else works from a recorded video.

## Test conventions

Known defects are pinned with `@pytest.mark.xfail(strict=True)` and a `reason` naming the defect. A normal run shows them as XFAIL. **When you fix one of these defects, delete its marker** — `strict=True` turns the now-passing test into an XPASS failure specifically to force that.

Two rules these gates are written to follow, because earlier versions violated both and could not distinguish a real fix from no fix:

- Assert through the public entry point, never against a private helper or module source text. A gate that calls an internal directly will keep failing after a call-site fix resolves the defect.
- Prefer `pytest.skip` with a reason over an assertion that passes vacuously (e.g. a `for` loop over a list that is empty because of the very defect under test).

`pytest --runxfail` shows what each gate actually asserts and is the fastest way to see whether a defect is still live.

Vision-model calls must be stubbed. `pipeline.py` does `from autojourney.analyser.llm import analyse_screen` at module scope, so patch `autojourney.pipeline.analyse_screen`, not the `analyser.llm` original.

Fixtures that exercise the stitcher need **smooth, non-repeating** texture (upscaled low-resolution noise works). Per-pixel noise gives Farneback optical flow nothing to track and `_detect_scroll_direction` misclassifies a vertical scroll as horizontal; a gradient repeats every 256 rows and leaves template matching several equally good alignments.

## Architecture

`run_pipeline()` in `pipeline.py` is the spine — it calls every stage in order and is the only place the full data flow is visible. The CLI (`cli/main.py`) is a thin `click` wrapper over it, plus `publish` and `report` subcommands that rehydrate a `JourneySession` from a previously written `session.json` and re-run just the tail of the pipeline.

The stage sequence and what crosses each boundary:

1. **capture** (`capture/agent.py`) — a video file or live USB stream becomes an iterator of `(frame_bgr, index, timestamp_ms)`. `save_frames` drains it to PNGs and returns a **manifest**: a plain `list[dict]` of `{"index", "timestamp_ms", "path"}`. This dict shape is an unmodelled contract between capture and the detector — it is not a dataclass.
2. **events** (`events/detector.py`) — `EventDetector.detect()` walks consecutive manifest entries and yields `FrameEvent`s using SSIM, changed-area fraction, and dense optical flow.
3. **stitch + screen collection** (inside `pipeline.py`) — this is where the event stream collapses into the screen list, and it is the least obvious stage. Only `SCROLL_END`, `TRANSITION`, and `MODAL` produce a `Screen`; `CONTENT_UPDATE` is detected but discarded, and `VIDEO_END` only backfills when nothing else produced a screen. Edges are created here, chaining each new screen to the previous one.
4. **analyse** (`analyser/llm.py`) — each screen image goes to a vision model and the parsed JSON populates `app_name`, `screen_name`, `ui_elements`, `inferred_action`, `probable_destinations`. All providers go through the `openai` client with `base_url` repointed; LM Studio is the default.
5. **graph** (`flow/graph.py`) — screens and edges become an `nx.DiGraph` plus a separate `{node_id: (x, y)}` layout dict. The layout tries graphviz, then a hand-rolled hierarchy walk, then spring layout.
6. **figma** (`figma/publisher.py`) — see below. Skipped unless `FIGMA_FILE_KEY` is set.
7. **report** (`report/markdown.py`) — writes `journey-report.md` from the session.

Outputs land under `OUTPUT_DIR` (default `./output`): `frames/`, `frames/manifest.json`, `events.json`, `screens/`, `session.json`, `journey-report.md`. `session.json` is the resume point for the `publish` and `report` subcommands.

### Shared models

`models.py` holds the dataclasses every stage passes around: `FrameEvent`, `Screen`, `FlowEdge`, `JourneySession`. `EventType` subclasses `str`, so it JSON-serialises to its value with no custom encoder — but note the `publish`/`report` CLI paths rebuild `Screen.event_type` from raw strings rather than `EventType` members.

### Config

`config.py` reads `.env` at **import time** into module-level constants. Setting an environment variable after import has no effect, so anything that needs different settings must take them as explicit arguments — which is why `EventDetector.__init__` accepts threshold overrides.

### Figma publishing

Publishing does **not** go through Figma's MCP server. The remote server's OAuth dynamic-client-registration endpoint 403s for any client not listed in Figma's MCP Catalog (AutoJourney isn't), and the local desktop MCP server doesn't expose `use_figma`/`upload_assets` — the only tools that can write canvas content. Neither path works.

Instead, `figma/publisher.py` drives an actual Figma Plugin at `autojourney/figma_plugin/` (`manifest.json`, `code.js`, `ui.html`), run manually by the user inside their own Figma session — no OAuth, no catalog gate. The split exists because of a hard Plugin API restriction: `code.js` (the main sandbox) has full document/scene-graph access but no network access; `ui.html` (an iframe shown via `figma.showUI`) has network access but no document access. So:

- `publisher.py` builds a plain-JSON spec (`_build_spec`) and starts `figma/plugin_server.py`'s `PluginBridgeServer` — a local `HTTPServer` serving `GET /spec`, `GET /image/<screen_id>`, and receiving `POST /progress` / `POST /complete`.
- `ui.html` fetches `/spec` and each screen's image bytes, then `postMessage`s them to `code.js`.
- `code.js` creates the page/frames/labels, calls `figma.createImage(bytes)` to place each screenshot directly (no separate upload-URL round trip), draws connectors, and posts `progress`/`complete` back through `ui.html` to the bridge server.
- `publish_to_figma()` blocks on `PluginBridgeServer.wait_for_completion()` until the plugin posts `/complete` (or times out).

The bridge server's port (`PLUGIN_BRIDGE_PORT` in `plugin_server.py`) must stay in sync with `figma_plugin/manifest.json`'s `networkAccess.devAllowedDomains` — it's a static asset, not templated. Connector nodes are FigJam-only, so screen-to-screen edges are drawn as arrow-capped vector paths instead.

`figma_plugin/` ships as package data (`[tool.setuptools.package-data]` in `pyproject.toml`) — its manifest path is resolved at runtime via `importlib.resources.files("autojourney.figma_plugin")` and printed for the user's one-time "Import plugin from manifest" step.
