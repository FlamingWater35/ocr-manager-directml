"""The app's runtime environment (Qt-free, stdlib + numpy only): the data
dir, the model cache, and the portable bundle.

Windows only. There is no engine installer any more: RapidOCR and
`onnxruntime-directml` are ordinary dependencies of the build, so the OCR
engine is present as soon as the app starts. What is left here is where the
app keeps its own files and whether it runs from a bundle.

| Module   | Role |
|----------|------|
| `paths`  | data dir, model cache, log file, `$OCR_MANAGER_BUNDLE` and bundle.json |
| `logfile`| sending stdout/stderr to a log file when there is no console |
| `logsetup`| the logging handlers: a file at DEBUG, plus the console |

`videocr/ocr_engine.py` is the engine itself; nothing here imports it.
"""
from __future__ import annotations

from pathlib import Path

from core.runtime.paths import Bundle, data_dir, host_arch, host_os, models_dir, supported_host

__all__ = [
    "Bundle",
    "data_dir",
    "host_arch",
    "host_os",
    "models_dir",
    "supported_host",
]


def prepare_data_dir(data: Path | None = None) -> Path:
    """Create the data and model directories, returning the data dir.

    Called once at startup, before any OCR engine is built: RapidOCR writes
    the recognition models it downloads into `models_dir()`, and a bundle's
    site-packages must not be written to.
    """
    root = data_dir() if data is None else Path(data)
    models_dir(root).mkdir(parents=True, exist_ok=True)
    return root
