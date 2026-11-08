"""What runs before anything else in `python -m app` / main.py (Qt-free).

1. Logging: `core.runtime.logsetup.configure()` gives the root logger a file
   handler on `<data>/logs/ocr-manager.log` at DEBUG (plus the console when
   there is one), so every `logger.debug()` in app/, core/ and videocr/ is
   written somewhere. Before this ran first, nothing configured logging and
   all of them were dropped.
2. stdout/stderr: under pythonw.exe (no console) there is nowhere for output
   to go, and `videocr.utils.suppress_output()` needs real file descriptors 1
   and 2; `core.runtime.logfile` sends both to the same log file (previous
   run kept as `.log.1`). It runs after the logging setup, because it rotates
   that file and a handler holding it open would make the rename fail on
   Windows.
3. The data dir and the model cache, and the bundle
   (`$OCR_MANAGER_BUNDLE`) when there is one.

There is no engine to activate: RapidOCR and onnxruntime-directml are
dependencies of the build, so importing `videocr.ocr_engine` is all an
installed engine ever needed. In developer mode the same dependencies come
from the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.runtime import logsetup, prepare_data_dir
from core.runtime.logfile import needs_redirect, redirect_if_needed
from core.runtime.paths import Bundle, BundleError, current_bundle, log_file


@dataclass
class Boot:
    bundle: Bundle | None = None
    bundle_error: str | None = None
    data: Path | None = None                # the data dir, created at boot
    log_path: Path | None = None            # where stdout/stderr went, when redirected
    log_config: Path | None = None          # where logging records go, when configured

    @property
    def bundled(self) -> bool:
        return self.bundle is not None


def boot() -> Boot:
    result = Boot()
    try:
        result.bundle = current_bundle()
    except BundleError as exc:
        result.bundle_error = str(exc)
    discard_devnull = result.bundle is not None
    has_console = not needs_redirect(discard_devnull=discard_devnull)
    result.log_path = redirect_if_needed(log_file(), discard_devnull=discard_devnull)
    result.log_config = logsetup.configure(log_file(), console=has_console)
    try:
        result.data = prepare_data_dir()
    except OSError:
        # A data dir that cannot be created (a read-only profile) must not
        # stop the window from opening: models fall back to the bundled copy
        # and anything that needs the cache reports its own error.
        result.data = None
    return result
