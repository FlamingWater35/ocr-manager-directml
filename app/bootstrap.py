"""What runs before anything else in `python -m app` / main.py (Qt-free).

1. stdout/stderr: under pythonw.exe (no console) there is nowhere for output
   to go, and `videocr.utils.suppress_output()` needs real file descriptors 1
   and 2; `core.runtime.logfile` sends both to
   `<data>/logs/ocr-manager.log` (previous run kept as `.log.1`).
2. The data dir and the model cache, and the bundle
   (`$OCR_MANAGER_BUNDLE`) when there is one.

There is no engine to activate: RapidOCR and onnxruntime-directml are
dependencies of the build, so importing `videocr.ocr_engine` is all an
installed engine ever needed. In developer mode the same dependencies come
from the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.runtime import prepare_data_dir
from core.runtime.logfile import redirect_if_needed
from core.runtime.paths import Bundle, BundleError, current_bundle, log_file


@dataclass
class Boot:
    bundle: Bundle | None = None
    bundle_error: str | None = None
    data: Path | None = None                # the data dir, created at boot
    log_path: Path | None = None            # where stdout/stderr went, when redirected

    @property
    def bundled(self) -> bool:
        return self.bundle is not None


def boot() -> Boot:
    result = Boot()
    try:
        result.bundle = current_bundle()
    except BundleError as exc:
        result.bundle_error = str(exc)
    result.log_path = redirect_if_needed(log_file(), discard_devnull=result.bundle is not None)
    try:
        result.data = prepare_data_dir()
    except OSError:
        # A data dir that cannot be created (a read-only profile) must not
        # stop the window from opening: models fall back to the bundled copy
        # and anything that needs the cache reports its own error.
        result.data = None
    return result
