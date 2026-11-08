"""The headless command-line modes of `python -m app` / main.py (CI hooks).

`--self-test`
    Imports every module under app/, core/ and videocr/; runs `ffmpeg
    -version` and `ffprobe -version` from PATH; calls
    `videocr.ffmpeg_capture.assert_reference_backend()`; reports the OCR
    device (`app.engine.engine_summary`). Prints one JSON report (`"ok"`
    plus the details). Exit 0 when everything passed, else 1.

`--ocr-smoke`
    Builds the OCR engine on the resolved device, OCRs a rendered line of
    Chinese (Latin when no CJK font is installed) and prints `RESULT
    <json>`. Exit 0 when any text was recognised, else 1.

`--download-models`
    Fetches the recognition models for the languages this build supports
    beyond the bundled PP-OCRv6 set (about 77 MB), printing `[model] <key>`
    lines and `RESULT <json>`. Exit 0 fetched or already present; 1 failed;
    130 interrupted. The app downloads them on first use instead; this is
    for preparing a machine ahead of time (CI, an offline install).

Output goes to stdout (the log file under pythonw).
"""
from __future__ import annotations

import importlib
import json
import os
import pkgutil
import platform
import subprocess
import sys
import threading
import time
import traceback

from app.bootstrap import Boot
from core import ffmpeg as ffmpeg_tool
from core.proc import TEXT_ENCODING, hidden_child
from app.version import __version__

PACKAGES = ("app", "core", "videocr")
TOOLS = ("ffmpeg", "ffprobe")
TOOL_TIMEOUT = 30
SMOKE_TEXT_CJK = "你好世界测试字幕"
SMOKE_TEXT_LATIN = "OCR SMOKE 12345"
PREFERRED_CJK_FONTS = ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "PingFang SC",
                       "WenQuanYi Micro Hei", "SimHei", "Hiragino Sans GB", "Noto Sans SC")
EXIT_OK, EXIT_FAIL, EXIT_INTERRUPTED = 0, 1, 130
_smoke_app = None                       # the QGuiApplication --ocr-smoke renders with, when none exists


def _emit(line: str) -> None:
    """Print one report line, never failing on the console's encoding.

    The reports carry CJK (the smoke text, and recognised lines from a real
    run) and Windows consoles are cp1252/cp437 by default, where printing
    them raises UnicodeEncodeError.  A CI step parsing this output cares about
    the bytes, so write UTF-8 and replace only what the stream truly cannot
    take.
    """
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            print(line.encode(encoding, "backslashreplace").decode(encoding), flush=True)
            return
        stream.flush()
        buffer.write((line + "\n").encode("utf-8"))
        buffer.flush()


# --------------------------------------------------------------------------
# --self-test
# --------------------------------------------------------------------------

def import_all(packages=PACKAGES) -> tuple[list[str], dict[str, str]]:
    """Import every module of `packages`; (imported, {module: error})."""
    imported, failed = [], {}
    for name in packages:
        try:
            package = importlib.import_module(name)
        except Exception as exc:                            # noqa: BLE001 - reported, not raised
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        imported.append(name)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{name}."):
            try:
                importlib.import_module(info.name)
                imported.append(info.name)
            except Exception as exc:                        # noqa: BLE001
                failed[info.name] = f"{type(exc).__name__}: {exc}"
    return imported, failed


def check_tool(tool: str) -> dict:
    path, origin = ffmpeg_tool.find_tool(tool)
    if path is None:
        return {"ok": False, "path": None, "origin": None,
                "error": f"{tool} is not in the bundle, the checkout or PATH"}
    try:
        done = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=TOOL_TIMEOUT,
                              check=False, **TEXT_ENCODING, **hidden_child())
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "path": path, "error": str(exc)}
    first = (done.stdout or "").splitlines()[0] if done.stdout else ""
    return {"ok": done.returncode == 0, "path": path, "origin": origin, "version": first,
            **({} if done.returncode == 0 else {"error": f"exit code {done.returncode}"})}


def check_decoder() -> dict:
    """The video decoder this build depends on: ffmpeg + ffprobe on PATH.

    There is no second backend to fall back to any more, so a missing binary
    is a hard failure here rather than a warning.
    """
    try:
        from videocr import ffmpeg_capture
    except Exception as exc:                                # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report = {
        "ok": bool(ffmpeg_capture.FFMPEG_AVAILABLE),
        "ffmpeg": ffmpeg_capture.FFMPEG or "",
        "ffprobe": ffmpeg_capture.FFPROBE or "",
        "backend": ffmpeg_capture.capture_backend_name(),
    }
    try:
        ffmpeg_capture.assert_reference_backend()
        report["reference_backend"] = True
    except Exception as exc:                                # noqa: BLE001
        report.update(ok=False, reference_backend=False, error=f"{type(exc).__name__}: {exc}")
    return report


def engine_report(boot: Boot) -> dict:
    from app.engine import engine_summary
    report: dict = {"bundled": boot.bundled}
    if boot.bundled:
        report["root"] = str(boot.bundle.root)
    report.update(engine_summary())
    return report


def self_test(boot: Boot) -> int:
    imported, failed = import_all()
    tools = {tool: check_tool(tool) for tool in TOOLS}
    decoder_report = check_decoder()
    ok = (not failed and not boot.bundle_error
          and all(t["ok"] for t in tools.values()) and decoder_report["ok"])
    report = {
        "ok": ok,
        "version": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "bundle": None if boot.bundle is None else {
            "root": str(boot.bundle.root), "version": boot.bundle.version,
            "os": boot.bundle.os, "arch": boot.bundle.arch},
        "bundle_error": boot.bundle_error,
        "imports": {"count": len(imported), "failed": failed},
        "tools": tools,
        "decoder": decoder_report,
        "engine": engine_report(boot),
    }
    _emit(json.dumps(report, indent=2, ensure_ascii=False))
    return EXIT_OK if ok else EXIT_FAIL


# --------------------------------------------------------------------------
# --download-models
# --------------------------------------------------------------------------

def download_models(boot: Boot) -> int:
    from core.runtime.paths import models_dir
    from videocr import ocr_engine

    target = models_dir(boot.data)
    cancel = threading.Event()
    box: dict = {}

    def progress(index: int, total: int, key: str) -> None:
        _emit(f"[model] {index + 1}/{total} {key}")

    def work() -> None:
        try:
            box["fetched"] = ocr_engine.download_models(target, progress=progress, cancel=cancel)
        except BaseException as exc:                         # noqa: BLE001
            box["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=work, name="model-download")
    thread.start()
    interrupted = False
    while thread.is_alive():
        try:
            thread.join(0.5)
        except KeyboardInterrupt:
            interrupted = True
            _emit("[step] Cancelling")
            cancel.set()
    if "error" in box:
        _emit(f"RESULT {json.dumps({'ok': False, 'error': box['error']})}")
        return EXIT_INTERRUPTED if interrupted else EXIT_FAIL
    fetched = box.get("fetched") or []
    _emit(f"RESULT {json.dumps({'ok': True, 'dir': str(target), 'fetched': fetched})}")
    return EXIT_INTERRUPTED if interrupted else EXIT_OK


# --------------------------------------------------------------------------
# --ocr-smoke
# --------------------------------------------------------------------------

def _cjk_family() -> str | None:
    from PyQt6.QtGui import QFontDatabase

    families = QFontDatabase.families(QFontDatabase.WritingSystem.SimplifiedChinese)
    for name in PREFERRED_CJK_FONTS:
        if name in families:
            return name
    return families[0] if families else None


def render_line(text: str | None = None):
    """(BGR numpy image, the text drawn, the font family) -- white text on
    black, drawn with Qt so any installed CJK font works."""
    import numpy as np
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter

    global _smoke_app
    if QGuiApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if sys.platform == "win32" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            # The offscreen platform looks for fonts in Qt's own folder, which
            # the wheel does not ship; Windows keeps them here.
            os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
        _smoke_app = QGuiApplication([sys.argv[0] if sys.argv else "ocr-manager"])
    family = _cjk_family()
    if text is None:
        text = SMOKE_TEXT_CJK if family else SMOKE_TEXT_LATIN
    width, height = 1280, 128
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(QColor("black"))
    painter = QPainter(image)
    font = QFont(family) if family else QFont()
    font.setPixelSize(64)
    painter.setFont(font)
    painter.setPen(QColor("white"))
    painter.drawText(image.rect(), int(Qt.AlignmentFlag.AlignCenter), text)
    painter.end()
    buffer = image.constBits()
    buffer.setsize(image.sizeInBytes())
    rgb = np.frombuffer(buffer, np.uint8).reshape(height, image.bytesPerLine())[:, :width * 3]
    rgb = rgb.reshape(height, width, 3)
    if rgb.any():
        return np.ascontiguousarray(rgb[:, :, ::-1]), text, family or QFont().family()
    # Qt found no usable font (a headless platform without a font folder):
    # draw the line with Pillow from a system font file instead.
    return _render_with_pillow(width, height)


# System font files Pillow can draw the smoke line with: (path, has CJK).
FONT_FILES = (
    (r"C:\Windows\Fonts\msyh.ttc", True), (r"C:\Windows\Fonts\simhei.ttf", True),
    (r"C:\Windows\Fonts\simsun.ttc", True), (r"C:\Windows\Fonts\arial.ttf", False),
)


def _render_with_pillow(width: int, height: int):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    for path, cjk in FONT_FILES:
        if os.path.exists(path):
            font = ImageFont.truetype(path, 64)
            text = SMOKE_TEXT_CJK if cjk else SMOKE_TEXT_LATIN
            break
    else:
        raise RuntimeError("no font to draw the smoke-test line with")
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (right - left)) / 2 - left, (height - (bottom - top)) / 2 - top), text,
              font=font, fill="white")
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1]), text, os.path.basename(path)


def ocr_smoke(boot: Boot) -> int:
    started = time.monotonic()
    try:
        image, expected, family = render_line()
        from videocr import utils

        device = utils.resolve_device(True)
        ocr = utils.create_ocr_engine("ch", None, None, True)
        texts, scores = [], []
        for item in ocr.predict(image):
            texts.extend(str(text) for text in item.get("rec_texts", []))
            scores.extend(float(score) for score in item.get("rec_scores", []))
    except Exception as exc:                                # noqa: BLE001
        traceback.print_exc()
        _emit(f"RESULT {json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})}")
        return EXIT_FAIL
    recognized = "".join(texts).strip()
    report = {"ok": bool(recognized), "device": device, "font": family, "expected": expected,
              "recognized": recognized, "scores": [round(score, 4) for score in scores],
              "exact": recognized.replace(" ", "") == expected.replace(" ", ""),
              "seconds": round(time.monotonic() - started, 1)}
    _emit(f"RESULT {json.dumps(report, ensure_ascii=False)}")
    return EXIT_OK if recognized else EXIT_FAIL
