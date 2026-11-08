"""Where ffmpeg and ffprobe come from.

Every frame the app reads is decoded by piping it out of ffmpeg, so ffmpeg is
not an optional extra a user installs: it ships with the app.  The packager
puts both binaries in the bundle's ``bin/`` (see packaging/build.py), the
launcher prepends that folder to PATH, and this module finds them whether or
not the launcher did.

Resolution order, first hit wins:

1. ``$OCR_MANAGER_FFMPEG_DIR`` -- an explicit override, for pointing a
   checkout at a build somewhere else.
2. ``<bundle>/bin`` -- a portable or installed tree, where ``bundle.json``
   names ``<bundle>`` and the sources live in ``<bundle>/src``.
3. ``<checkout>/bin`` -- running from source, next to main.py.
4. ``PATH`` -- last resort, for a developer machine with ffmpeg installed
   system-wide but no copy in the tree.

A miss is not fatal at import: the app starts, reports the missing tool
(``app.main_window`` banners it, ``--self-test`` fails on it) and refuses to
decode rather than pretending.  Nothing here imports Qt.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from core.runtime.paths import source_root

# Point this at a folder holding ffmpeg/ffprobe to use those instead.
FFMPEG_DIR_ENV = "OCR_MANAGER_FFMPEG_DIR"

TOOL_NAMES = ("ffmpeg", "ffprobe")

# Where each tool was found, for --self-test and the banner.
ORIGIN_ENV = "env"
ORIGIN_BUNDLE = "bundle"
ORIGIN_CHECKOUT = "checkout"
ORIGIN_PATH = "path"


def _exe(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def _bundle_root() -> Path | None:
    """The bundle named by $OCR_MANAGER_BUNDLE, if there is one."""
    raw = os.environ.get("OCR_MANAGER_BUNDLE")
    if not raw:
        return None
    root = Path(raw)
    return root if (root / "bundle.json").is_file() else None


def _candidate_dirs() -> list[tuple[str, Path]]:
    """(origin, directory) pairs to look in, in priority order."""
    dirs: list[tuple[str, Path]] = []
    override = os.environ.get(FFMPEG_DIR_ENV)
    if override:
        dirs.append((ORIGIN_ENV, Path(override)))

    bundle = _bundle_root()
    if bundle is not None:
        dirs.append((ORIGIN_BUNDLE, bundle / "bin"))

    try:
        root = source_root()
    except OSError:                      # a zipimported or relocated tree
        root = None
    if root is not None:
        # <checkout>/bin when running from source; <bundle>/bin from
        # <bundle>/src when the bundle was not named by the environment.
        dirs.append((ORIGIN_CHECKOUT, root / "bin"))
        dirs.append((ORIGIN_CHECKOUT, root.parent / "bin"))
    return dirs


def find_tool(name: str) -> tuple[str | None, str | None]:
    """(path, origin) for ``name``; (None, None) when it is not anywhere.

    ``path`` is absolute.  The bundled copies win over PATH so a machine with
    an older system ffmpeg decodes with the build the app was tested against.
    """
    exe = _exe(name)
    seen: set[str] = set()
    for origin, directory in _candidate_dirs():
        candidate = directory / exe
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate), origin

    found = shutil.which(name)
    if found is not None:
        return found, ORIGIN_PATH
    return None, None


def ffmpeg_path() -> str | None:
    return find_tool("ffmpeg")[0]


def ffprobe_path() -> str | None:
    return find_tool("ffprobe")[0]


def ffmpeg_dir() -> str | None:
    """The folder holding both binaries, when both were found in one place.

    Used to put them on PATH for a child process that looks tools up by name
    (the test suite does), without moving the ones already resolved.
    """
    ffmpeg = ffmpeg_path()
    ffprobe = ffprobe_path()
    if ffmpeg is None or ffprobe is None:
        return None
    folder = os.path.dirname(os.path.abspath(ffmpeg))
    if os.path.dirname(os.path.abspath(ffprobe)) != folder:
        return None
    return folder


def ffmpeg_argv(*args: str) -> list[str]:
    """The argv that runs ffmpeg with ``args``.

    Callers that can run without ffmpeg check :func:`ffmpeg_path` first; this
    falls back to the bare name so a child process reports the miss itself.
    """
    return [ffmpeg_path() or "ffmpeg", *args]


def ffprobe_argv(*args: str) -> list[str]:
    return [ffprobe_path() or "ffprobe", *args]


def available() -> bool:
    return ffmpeg_path() is not None and ffprobe_path() is not None


def missing_tools() -> list[str]:
    return [name for name in TOOL_NAMES if find_tool(name)[0] is None]


def describe() -> dict:
    """What --self-test reports: the path and origin of each tool."""
    report = {}
    for name in TOOL_NAMES:
        path, origin = find_tool(name)
        report[name] = {"path": path, "origin": origin, "ok": path is not None}
    return report


def where_we_looked() -> list[str]:
    """The directories searched, for an error message that can be acted on."""
    return [str(directory) for _origin, directory in _candidate_dirs()] + ["PATH"]


def missing_message() -> str:
    """What to tell a user whose install is missing the decoder."""
    names = ", ".join(missing_tools())
    return (f"{names} could not be found. They ship with OCR Manager in its "
            f"bin folder; reinstall the app, or set {FFMPEG_DIR_ENV} to a "
            f"folder holding them.")
