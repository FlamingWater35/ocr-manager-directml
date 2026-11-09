# AGENTS.md

Guidance for agents (and people) working in this repository. Read it before
changing code; most of the rules below are enforced by tests that will fail
if you break them, and the ones that are not are still load-bearing.

## What this is

**OCR Manager** is a Windows desktop app (PyQt6) that reads hardsubbed
subtitles out of video and writes timed `.ass` files. It decodes with a
bundled `ffmpeg` subprocess, OCRs with **RapidOCR** on **ONNX Runtime
DirectML**, and detects per-folder crop boxes, brightness thresholds and keep
ranges automatically.

Three things are non-negotiable, and they define the shape of the codebase:

1. **Windows x64 only.** There is no Linux or macOS build, no per-OS branch
   in the packaging, and no "port it back" scaffolding. `core/runtime/paths.py`
   hard-codes `HOST_OS = "win"` / `HOST_ARCH = "x86_64"`; its only
   `sys.platform` use is `supported_host()` reporting whether this process is
   the one the build targets.
2. **No CUDA, no PaddleOCR, no PyAV.** The engine is RapidOCR on
   `onnxruntime-directml` (falls back to CPU); video is decoded by piping
   ffmpeg's raw BGR frames; every frame's PTS is read from ffmpeg's
   `showinfo` output rather than estimated from an ordinal. Do not reintroduce
   a Python video binding or a second FFmpeg build — `videocr/ffmpeg_capture.py`
   explains why the last one was removed.
3. **ffmpeg ships with the app.** Nothing is installed on first run. The
   binaries live in `bin/` next to the launcher and are resolved through
   `core/ffmpeg.py`, never by bare name from application code.

## Commands

```powershell
# Setup (Windows, Python 3.11 or 3.12)
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
# then put ffmpeg.exe + ffprobe.exe in .\bin  (see README.md — take the GPL build)

# Run
.venv\Scripts\python main.py [folder]        # or: python -m app

# Tests
$env:QT_QPA_PLATFORM = "offscreen"; .venv\Scripts\python -m pytest tests\ -m "not slow"
.venv\Scripts\python -m pytest tests\ui                       # needs the offscreen platform too
.venv\Scripts\python -m pytest tests\test_detect_crop.py -k crop_box

# Headless checks (these are what CI runs against the built bundle)
.venv\Scripts\python -m app --self-test        # JSON report: device, decoder, tools, logging
.venv\Scripts\python -m app --ocr-smoke        # renders one line and reads it back
.venv\Scripts\python -m app --download-models  # prefetch every language model + tiny/medium sizes
.venv\Scripts\python -m app --version

# Golden OCR output, and the release build
.venv\Scripts\python tools\fidelity_check.py
.venv\Scripts\python packaging\build.py --bundle-only
```

`QT_QPA_PLATFORM=offscreen` is required for anything that constructs a widget
— there is no display in CI, and pytest-qt is deliberately **not** installed
(`tests/ui/conftest.py` supplies the one session-scoped `QApplication` fixture
instead of `qtbot`).

## Layout

```
app/            the PyQt6 window — the only layer that may import Qt
  __main__.py     argument parsing, boot(), then MainWindow
  bootstrap.py    boot order: bundle → console/log redirect → logging → data dir
  cli.py          --self-test / --ocr-smoke / --download-models
  controller.py   ProjectController: the one object views talk to
  engine.py       engine_problem() / decoder_problem() for the startup banner
  views/          the workbench UI (top bar, queue, stage, inspector, tabs, episode screens)
  widgets/base.py, theme/{tokens,qss}.py    the design system
core/           Qt-free domain: detectors, jobs, project model, runtime paths
  ffmpeg.py       the one place that decides which ffmpeg/ffprobe to use
  proc.py         hidden_child() / TEXT_ENCODING for every child process
  detect/         crop, brightness, confirm, lines, vad, ocr_view, ranges/
  jobs/           runner (lanes/events), detect_jobs, run, apply, autopilot, view_cache
  project/        model, store, migrate, layout, languages, ocr_kwargs
  runtime/        paths, logfile, logsetup
videocr/        Qt-free OCR pipeline: capture, engine, engine_registry, video, api, label_scanner
packaging/      build.py (Windows bundle + installer), installer.iss, launcher.c
tools/          fidelity_check.py (goldens), bench.py
tests/          58 modules + tests/ui/ 16 modules + tests/fixtures/
```

## Architecture rules

These are checked by tests. Breaking one turns a green suite red, so treat the
test as the specification and read it before you work around it.

- **Views never import `core` or `videocr`.** `app/views/**` and
  `app/main_window.py` go through the non-view app modules instead
  (`app.controller`, `app.engine`, `app.imaging`, `app.masking`,
  `app.state_text`, …), which may import `core`. A bare `import videocr` is
  the single exception the check allows.
  → `tests/ui/test_main_window.py::test_views_import_no_core_modules`
- **`core/` and `videocr/` import no Qt.** Several tests spawn a subprocess and
  assert no `PyQt6`/`PyQt5`/`PySide6` module got loaded. If you need Qt in a
  domain module, you need a callback or a plain dataclass instead.
  → `test_detect_jobs.py::test_job_modules_import_no_qt`,
  `test_view_jobs.py::test_view_jobs_import_no_qt`, plus the same check in
  `test_run_job.py`, `test_view_cache.py`, `test_autopilot.py`
- **`core/project` imports no PyQt6** at all, by design — it is the shared
  model.
- **`ProjectController` is the only code that mutates the model**, and it does
  so on the GUI thread. Views read through it and call its commands; job events
  reach it through a `queue.SimpleQueue` drained by a 30 ms `QTimer`.
- **Every child process** passes `**hidden_child()` (no console window flashing
  up under `pythonw.exe`) and, in text mode, `**TEXT_ENCODING` (UTF-8, because
  ffprobe's JSON carries CJK file names and the locale is cp1252).
  → `tests/test_cross_platform.py::test_every_child_process_is_started_hidden_and_text_is_read_as_utf8`
- **Never call `ffmpeg`/`ffprobe` by bare name** from `app/`, `core/` or
  `videocr/`. Use `core.ffmpeg.ffmpeg_argv(...)` / `ffprobe_argv(...)` (which
  fall back to the bare name so a child reports the miss itself) or
  `ffmpeg_path()` / `ffprobe_path()`. → `tests/test_ffmpeg_resolution.py`
- **Detection never overrides the user.** An apply writes only into an empty
  field or over a value whose source is `DETECTED`/`HINT`; `MANUAL` and
  `IMPORTED` are never overwritten. A doubted result is stored *with* its
  blocking flag so the file goes to review with something to accept.
  → `core/jobs/apply.py`'s module docstring
- **`tests/goldens/*.ass` must stay byte-identical.** Any change that alters
  OCR output is a fidelity regression until proven otherwise:
  `tools/fidelity_check.py`. ASS output is LF on every OS.

## Conventions

- **Docstrings carry the reasoning.** Modules start with a docstring that
  states the invariants and often cites the test that pins them; the large
  tables in `core/detect/__init__.py` (frame addressing, cancellation, flags)
  are the reference for re-fetching a frame or cancelling a detector — read
  them before writing a detector or a preview.
- **No linter or formatter is configured** (no ruff/black/mypy/isort config,
  and none installed in CI). Match the surrounding code: 4-space indent,
  `from __future__ import annotations` in newer modules, `logger =
  logging.getLogger(__name__)`. There is no hard line-length limit — long
  explanatory comments and table rows run past 100 columns on purpose.
- **Logging is real.** `core/runtime/logsetup.py` configures a DEBUG file
  handler at boot (`%LOCALAPPDATA%\ocr-manager\logs\ocr-manager.log`, rotated
  at 8 MB); nothing is dropped. Add `logger.debug` lines with the values that
  would answer "why did this decide that?" — the crop-probe score line exists
  precisely because auto-crop silently found nothing once.
- **Errors that the user must see go in the UI banner** (`app/engine.py`
  returns `(title, body)` tuples); everything else is logged. Never let a
  failed apply or a missing tool raise into a Qt slot — PyQt6 aborts the
  process, which is why `app/__main__.py` installs an `sys.excepthook`.
- **Commit messages** follow conventional-ish prefixes (`feat:`, `fix(ui):`,
  `perf:`, `test:`, `ci:`, `chore: version X`); large changes are plain
  sentences. `app/version.py` is the single source of the version, and a
  `vX.Y.Z` tag that does not match it fails CI.

## Tests

- `pytest.ini` sets `testpaths = tests` and a 900 s timeout. Markers:
  `needs_media` (needs the reference projects in `tests/fixtures/media_manifest.json`),
  `slow` (>30 s), `timing` (wall-clock budget — never in a parallel run).
- **Test modules import each other** (`from test_detect_crop import ...`).
  This works because `tests/` has no `__init__.py`, so pytest inserts it on
  `sys.path`. Do not add `tests/__init__.py`.
- `tests/conftest.py` gives every test an empty OCR engine pool
  (`_reset_engine_registry`), its own episode cache root
  (`OCR_MANAGER_CACHE_DIR`), and — session-scoped, so clip fixtures encode
  after it — the app's own ffmpeg folder prepended to `PATH`.
- Media is synthesized with ffmpeg (`testsrc2`, `smptebars`, `drawtext`).
  Anything that needs the real reference projects skips cleanly instead of
  failing. `tests/decode_reference.py` is the independent, PyAV-free reference
  decode the label tests compare against.
- **The label tests are slow** (`test_label_frame_identity.py`,
  `test_label_boundaries.py`, `test_label_sampling.py`, `test_label_progress.py`):
  the phase-4 boundary sweep seeks with a fresh ffmpeg per seek, so the four
  modules together run for well over an hour. Run them deliberately, not in a
  quick loop.
- When a fake external tool is needed, write it as a **Python** script and
  point `ffmpeg_tool.ffmpeg_argv` at `[sys.executable, script, ...]`. A
  `#!/bin/sh` file cannot be started on Windows — `shutil.which` skips
  extensionless scripts, so the real binary would run instead and the test
  would silently pass for the wrong reason.

## Real-world test media (`test_media/`, git-ignored)

The project keeps two real 4K clips in `test_media/`, with a competitor's OCR
output beside each. **Read this before comparing any subtitle file in there.**

| File | What it is |
|---|---|
| `test_video.mkv` | 349.2 s, HEVC Main 10, 3840×1640, 25 fps, bt709 **SDR** (not HDR). Has an **embedded ASS track** (`language=eng`). |
| `test_video.ass` | 102 Chinese lines. **VideOCR Recreated's OCR output — not truth.** |
| `test_video_2.mp4` | 17.5 s, HEVC Main, 3840×1608. **No embedded subtitle stream.** |
| `test_video_2.zh.ass` | 11 Chinese lines (8 dialogue + 3 `{\pos(...)}` labels). VideOCR Recreated's output. |

**The embedded English track in `test_video.mkv` is the ground truth** for that
clip's text and timings — it was written by hand in Aegisub, so it is what a
human says is on screen. `test_video.ass` is *not* that track: it is a sample of
VideOCR Recreated's output, and the long clip's is known to have issues. Where
they disagree, the embedded track wins; a line VideOCR Recreated missed is a
recall failure to investigate, and a line it invented is a precision failure.
`test_video_2.zh.ass` is the short clip's only subtitle, so it is a sample of
expected label output rather than verified text.

Re-extract the reference track with
`bin\ffmpeg.exe -y -i test_media\test_video.mkv -map 0:s:0 -c:s copy out.ass`.
The tests that need these clips carry the `needs_real_media` marker and skip
when the folder is absent.

## Gotchas worth knowing before you debug

- **RapidOCR builds its sub-engines lazily.** `RapidEngine.providers` is `[]`
  right after construction; call `engine.warm_up()` (or run one inference)
  before asserting on the provider list. `DmlExecutionProvider` silently falls
  back to CPU, so verify the real list rather than trusting the flag.
- **Params values for `engine_type`/`model_type`/`ocr_version`/`task_type`
  must be Enum instances**, not strings, or `ParseParams.update_batch` raises
  `TypeError: The value of Det.model_type must be Enum Type.` Dotted keys are
  `Section.key` and Global keys must actually exist.
- **The tier must be in the engine-pool key** (`videocr/engine_registry.py`),
  or two folders set to different model sizes will share weights.
- **`seek_to_display_time(t)` may answer from the frame it just delivered**;
  it returns `None` (not `False`) when it cannot, which makes the caller do a
  real seek. Running out of seek retries means "landed late", not "past the
  end" — `seek_past_end` stays `False` so a forward scan continues.
- **A frame-history cache was tried twice and removed twice**: serving a
  backward seek from already-decoded frames changed 4 of 256 phase-4 boundary
  results. Exactness wins over speed here; do not re-add one without proving
  the boundary suite is unchanged.
- **`DT_SCORE_THRESHOLD` is calibrated for PP-OCRv6, not PaddleOCR.** The
  detector scores the same text far lower than Paddle's DB did (0.75–0.87 for
  real subtitle text). Raising it back toward 0.9 makes auto-crop find nothing.
- Paths from `QUrl.toLocalFile()` use forward slashes on Windows; compare with
  `os.path.join`, not `str(tmp_path) + "/x"`.

## Environment variables

| Variable | Effect |
|---|---|
| `OCR_MANAGER_DATA_DIR` | overrides `%LOCALAPPDATA%\ocr-manager` (models, logs) |
| `OCR_MANAGER_FFMPEG_DIR` | points the resolver at a different ffmpeg/ffprobe folder |
| `OCR_MANAGER_BUNDLE` | a portable build's root; only honoured when `<root>/bundle.json` exists |
| `OCR_MANAGER_CACHE_DIR` | episode cache root (tests set it per test) |
| `OCR_MANAGER_LOG_LEVEL` | `debug` (default), `info`, `warning`, `error`, `critical` |
| `OCR_MANAGER_UI_SCALE` | UI scale multiplier, clamped to 1.0–3.0 (default 1.25) |
| `QT_QPA_PLATFORM` | set to `offscreen` for tests and CI |

## Git notes

- `.gitignore` ignores `*.md` with exceptions for `!README.md` and
  `!AGENTS.md`; it also ignores the local-only `bin/`, `models/`, `.venv/`,
  `.tmp/`, `.build-cache/`. (`.conda/` carries its own nested `.gitignore`.)
  `bin/` and `models/` are fetched per checkout and must never be committed.
- `.git/info/exclude` carries a couple of local-only entries left by tests;
  leave it alone.
