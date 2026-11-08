"""The OCR engine as the window sees it: an app-level helper over
`videocr.ocr_engine`, so views never import `videocr` themselves.

There is nothing to install any more. RapidOCR and `onnxruntime-directml`
are dependencies of the build, so an engine exists as soon as the app
starts; what can still go wrong is that DirectML is unavailable (an old
Windows, a machine with no DX12 GPU, a CPU-only VM) or that a recognition
model for a non-default language cannot be downloaded. `engine_problem()`
is the main window's startup check for the first case, and
`engine_summary()` is what `--self-test` reports.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DIRECTML_HINT = (
    "OCR runs on the CPU. DirectML needs Windows 10 build 18362 or newer "
    "and a DirectX 12 capable graphics adapter."
)


def engine_summary() -> dict:
    """What device OCR will run on, for --self-test and the log.

    Never raises: a broken install is reported, not thrown, because this
    runs in the window's startup path.
    """
    try:
        from videocr import ocr_engine
    except Exception as exc:                                # noqa: BLE001 - reported
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        available = ocr_engine.dml_available()
    except Exception as exc:                                # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "device": ocr_engine.DML if available else ocr_engine.CPU,
        "directml": available,
        "note": "" if available else DIRECTML_HINT,
    }


def engine_problem() -> tuple[str, str] | None:
    """(title, text) when OCR cannot run at all; None otherwise.

    A missing DirectML provider is not a problem: OCR still works, on the
    CPU. Only a missing or unimportable engine is, and that is a broken
    installation rather than something the user can set up.
    """
    summary = engine_summary()
    if summary.get("ok"):
        return None
    return ("OCR engine unavailable",
            f"The OCR engine could not be loaded.\n\n{summary.get('error', '')}\n\n"
            f"Reinstall OCR Manager.")
