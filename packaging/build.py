#!/usr/bin/env python3
"""Build OCR Manager's Windows release artifacts.

    python packaging/build.py                 # bundle + artifacts into dist/
    python packaging/build.py --bundle-only   # just the runnable tree
    python packaging/build.py run -- --self-test   # run the built bundle (console)

Stdlib only; any Python 3.11+ runs it. What it builds:

    <root>/python/          python-build-standalone CPython with every runtime
                            dependency in its own site-packages, RapidOCR and
                            onnxruntime-directml included
    <root>/src/             main.py, app/, core/, videocr/, resources/
    <root>/bin/             static ffmpeg + ffprobe (put on PATH by the launcher)
    <root>/models/          PP-OCRv6 detector and recogniser, tiny and medium
                            (the wheel already carries small)
    <root>/licenses/        third-party notices
    <root>/bundle.json      {"version", "os", "arch"}

Nothing is installed on first run any more: the OCR engine is part of the
bundle, and so are all three PP-OCRv6 model sizes, so a fresh copy reads
subtitles immediately at whichever size a folder picks. The only thing fetched
at run time is a recognition model for a language PP-OCRv6 does not cover, into
the user's data folder (`--download-models` prefetches all of them).

Artifacts, named ocr-manager-v<X.Y.Z>-win.<ext>: .exe (per-user Inno Setup
installer) and .zip (portable tree).

Everything downloaded is pinned by URL and SHA-256 and cached in --cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import traceback
import zipfile
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGING = REPO / "packaging"

APP_NAME = "OCR Manager"
BUNDLE_ID = "uk.blyat.ocr-manager"

# python-build-standalone: https://github.com/astral-sh/python-build-standalone
PYTHON = "3.12.14"
PBS_RELEASE = "20260901"
PBS_ASSETS = {
    ("win", "x86_64"): ("x86_64-pc-windows-msvc",
                        "7c45c9622400d578709a9b2cddbe8124cc21d382409d9f13406d706d28e31b14"),
}

# Static ffmpeg/ffprobe: BtbN's LGPL build, a month-end autobuild (kept for two
# years). The app decodes video by piping frames out of this ffmpeg, so this
# build is what decides how the frames look.
_BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-31-13-27/"
FFMPEG = {
    "win": [(_BTBN + "ffmpeg-n8.1.2-50-g1a748fe2cd-win64-lgpl-8.1.zip",
             "f6274bbd9c247f9e90c1bbed066b03ed4a3907cece2fb91be6dd352393936365")],
}

# The PP-OCRv6 sizes the rapidocr wheel does not carry (it ships `small`
# only). Both the detector and the recogniser, because a folder set to tiny or
# medium reads with both. Same files and digests RapidOCR's own
# `default_models.yaml` names for v3.10.0.
_MODELSCOPE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/"
OCR_MODELS = [
    (_MODELSCOPE + "det/PP-OCRv6_det_tiny.onnx",
     "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5"),
    (_MODELSCOPE + "rec/PP-OCRv6_rec_tiny.onnx",
     "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b"),
    (_MODELSCOPE + "det/PP-OCRv6_det_medium.onnx",
     "92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2"),
    (_MODELSCOPE + "rec/PP-OCRv6_rec_medium.onnx",
     "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b"),
]

# The MSVC runtime the bundled wheels (onnxruntime, PySide6, OpenCV, scipy) link
# against, shipped app-local next to python.exe so no VC++ redistributable
# install is needed.
MSVC_RUNTIME = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
                "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll", "vcomp140.dll")

TAIL_LINES = 80

SOURCES = ("main.py", "app", "core", "videocr", "resources")

# The interpreter flags every launcher uses: no user site (-s) and no PYTHON*
# variables (-E), so nothing on the user's machine leaks in; no bytecode
# writes (-B: the tree may be read-only, and the installer may be running from
# a read-only place); UTF-8 mode.
PY_FLAGS = ("-s", "-E", "-B", "-X", "utf8")


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


class CommandFailed(Exception):
    """A build command exited non-zero; carries the end of its output."""


def stream(cmd: list, **kw) -> int:
    """Run `cmd`, echoing its output live and keeping the last lines: CI
    job logs need a token to read, so a failure is reported through a
    public annotation (see annotate()) that carries them."""
    if kw.get("stdout") is not None:
        return subprocess.run([str(c) for c in cmd], **kw).returncode
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", **kw)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line.rstrip())
    code = proc.wait()
    if code:
        stream.tail = list(tail)
    return code


stream.tail = []


def run(cmd: list, **kw) -> None:
    print("   $", " ".join(str(c) for c in cmd), flush=True)
    if code := stream(cmd, **kw):
        raise CommandFailed(f"exit code {code}: " + " ".join(str(c) for c in cmd)
                            + "\n" + "\n".join(stream.tail))


def annotate(title: str, text: str) -> None:
    """A GitHub Actions error annotation (readable without a token)."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    body = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title={title}::{body}", flush=True)


def host() -> tuple[str, str]:
    system, machine = platform.system(), platform.machine().lower()
    os_name = {"Windows": "win"}.get(system)
    arch = {"amd64": "x86_64", "x86_64": "x86_64"}.get(machine)
    if (os_name, arch) not in PBS_ASSETS:
        sys.exit(f"unsupported build host: {system} {machine} (Windows x64 only)")
    return os_name, arch


def app_version() -> str:
    text = (REPO / "app" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if not match:
        sys.exit("app/version.py has no __version__")
    return match.group(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, cache: Path, expected: str | None = None) -> Path:
    """Download `url` into `cache` once; verify its SHA-256 when one is given."""
    name = url.rstrip("/").rsplit("/", 1)[-1].replace("%2B", "+")
    dest = cache / name
    if dest.exists() and (expected is None or sha256(dest) == expected):
        return dest
    log(f"downloading {name}")
    cache.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ocr-manager-build"})
    with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    if expected is not None and (actual := sha256(part)) != expected:
        part.unlink()
        sys.exit(f"{name}: SHA-256 {actual} != pinned {expected}")
    os.replace(part, dest)
    return dest


# -- the runnable tree ------------------------------------------------------

def python_exe(root: Path, os_name: str, gui: bool = False) -> Path:
    if os_name == "win":
        return root / "python" / ("pythonw.exe" if gui else "python.exe")
    return root / "python" / "bin" / "python3"


def install_python(root: Path, os_name: str, arch: str, cache: Path) -> None:
    triple, digest = PBS_ASSETS[(os_name, arch)]
    name = f"cpython-{PYTHON}+{PBS_RELEASE}-{triple}-install_only_stripped.tar.gz"
    url = (f"https://github.com/astral-sh/python-build-standalone/releases/download/"
           f"{PBS_RELEASE}/{name.replace('+', '%2B')}")
    archive = fetch(url, cache, digest)
    log("unpacking the interpreter")
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="tar")          # the archive's top dir is python/
    lib = root / "python" / ("Lib" if os_name == "win" else f"lib/python{PYTHON.rsplit('.', 1)[0]}")
    for unused in ("test", "idlelib", "turtledemo"):
        shutil.rmtree(lib / unused, ignore_errors=True)


def install_dependencies(root: Path, os_name: str) -> None:
    py = python_exe(root, os_name)
    pip = [py, "-s", "-E", "-m", "pip"]
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_WARN_SCRIPT_LOCATION": "1"}
    log("installing the runtime dependencies")
    run([*pip, "install", "--upgrade", "pip"], env=env)
    run([*pip, "install", "--only-binary=:all:",
         "-r", PACKAGING / "requirements-bundle.txt",
         "-c", PACKAGING / "constraints-bundle.txt"], env=env)
    if os_name == "win":
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        for dll in MSVC_RUNTIME:
            if (system32 / dll).exists() and not (root / "python" / dll).exists():
                shutil.copy2(system32 / dll, root / "python" / dll)


# Qt modules the app never loads. It uses QtCore, QtGui and QtWidgets; the
# Windows platform plugin resolves to exactly those three (dumpbin on
# qwindows.dll / qmodernwindowsstyle.dll shows no other Qt dependency), so the
# rest of the wheel is dead weight. PySide6-Essentials ships all of Qt anyway
# -- 647 MB in site-packages, of which Qt6WebEngineCore.dll alone is 194 MB --
# and every name below is matched as a PREFIX of the module name, so "Quick"
# also takes Quick3D and QuickControls2.
#
# Kept on purpose, and why: Svg/SvgWidgets because the icon engine and the
# imageformats plugin link Qt6Svg (qsvgicon.dll, qsvg.dll), Network and
# OpenGL because Qt can load them lazily from a platform plugin, Xml and
# Concurrent because QtWidgets' own .pyd imports them. Anything not listed
# here stays.
QT_UNUSED = ("3D", "AxContainer", "Bluetooth", "CanvasPainter", "Charts", "DataVisualization",
             "Designer", "Graphs", "Help", "HttpServer", "Labs", "Location", "Lottie",
             "Multimedia", "NetworkAuth", "Nfc", "Pdf", "Positioning", "Qml", "Quick",
             "RemoteObjects", "Scxml", "Sensors", "SerialBus", "SerialPort", "ShaderTools",
             "SpatialAudio", "Sql", "StateMachine", "Test", "TextToSpeech", "UiTools",
             "VirtualKeyboard", "WebChannel", "WebEngine", "WebSockets", "WebView")

# Directories under site-packages/PySide6 that only a build, a designer session
# or a QML app reads. `qml` is 20 MB of QML types, `translations` 59 MB of Qt's
# own .qm files (the app installs no QTranslator, so they could never load),
# and `metatypes`/`typesystems`/`glue` are shiboken build inputs.
QT_UNUSED_DIRS = ("doc", "glue", "include", "lib", "metatypes", "qml", "scripts", "support",
                  "translations", "typesystems", "qsci")

# The designer/linguist/QML toolchain, which ships as loose .exes next to the
# libraries. QtWebEngineProcess.exe belongs to the WebEngine we just removed.
QT_UNUSED_EXES = ("assistant.exe", "balsam.exe", "balsamui.exe", "designer.exe", "linguist.exe",
                  "lrelease.exe", "lupdate.exe", "qmlcachegen.exe", "qmlformat.exe",
                  "qmlimportscanner.exe", "qmllint.exe", "qmlls.exe", "qmltyperegistrar.exe",
                  "qsb.exe", "QtWebEngineProcess.exe", "rcc.exe", "svgtoqml.exe", "uic.exe")

QT_UNUSED_PLUGINS = ("assetimporters", "canbus", "designer", "geometryloaders", "geoservices",
                     "help", "multimedia", "position", "qmltooling", "renderers",
                     "renderplugins", "sceneparsers", "scxmldatamodel", "sensors", "sqldrivers",
                     "texttospeech", "webview")

# One entry name, one module name: "Qt6Widgets.dll" and "QtWidgets.pyd" are both
# module "Widgets", which is what QT_UNUSED is written against.
_QT_MODULE = re.compile(r"^Qt6?([A-Za-z0-9_]+)\.(?:dll|pyd)$")


def qt_module(name: str) -> str | None:
    """The Qt module an entry provides, or None for anything else."""
    match = _QT_MODULE.match(name)
    return match.group(1) if match else None


def prune_qt(root: Path, os_name: str) -> None:
    """Drop the Qt the app cannot reach, and refuse to guess if the layout moved.

    PySide6 lays everything out flat (Qt6*.dll, Qt*.pyd, plugins/ and the tool
    .exes all in site-packages/PySide6), unlike PyQt6's PyQt6/Qt6/ nesting, so
    this walks one directory. If the three modules the app actually imports do
    not survive, that is a build failure rather than a quiet half-Qt bundle.
    """
    lib = root / "python" / ("Lib" if os_name == "win" else f"lib/python{PYTHON.rsplit('.', 1)[0]}")
    pyside = lib / "site-packages" / "PySide6"
    if not pyside.is_dir():
        raise CommandFailed(f"PySide6 is not in the bundle: {pyside} does not exist")
    unused = re.compile(r"^(?:%s)" % "|".join(QT_UNUSED))
    # Qt Multimedia's own FFmpeg libraries: the app decodes with bin/ffmpeg.exe.
    qt_ffmpeg = re.compile(r"^(?:avcodec|avformat|avutil|swresample|swscale)[-.]")
    removed = 0
    for entry in pyside.iterdir():
        module = qt_module(entry.name)
        if entry.is_file():
            if (module and unused.match(module)) or qt_ffmpeg.match(entry.name):
                entry.unlink()
                removed += 1
            elif entry.name == "opengl32sw.dll":
                # Mesa's software rasteriser: the app draws with QPainter and
                # never asks for OpenGL.
                entry.unlink()
                removed += 1
        elif entry.is_dir() and entry.name in QT_UNUSED_DIRS:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    for name in QT_UNUSED_EXES:
        target = pyside / name
        if target.exists():
            target.unlink()
            removed += 1
    for name in QT_UNUSED_PLUGINS:
        shutil.rmtree(pyside / "plugins" / name, ignore_errors=True)
    # WebEngine's runtime resources live in resources/, not next to the DLL.
    for name in ("qtwebengine_devtools_resources.debug.pak", "qtwebengine_devtools_resources.pak",
                 "qtwebengine_resources.debug.pak", "qtwebengine_resources.pak",
                 "qtwebengine_resources_100p.debug.pak", "qtwebengine_resources_100p.pak",
                 "qtwebengine_resources_200p.debug.pak", "qtwebengine_resources_200p.pak",
                 "icudtl.dat", "v8_context_snapshot.bin", "v8_context_snapshot.debug.bin"):
        target = pyside / "resources" / name
        if target.exists():
            target.unlink()
            removed += 1
    for required in ("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll",
                     "QtCore.pyd", "QtGui.pyd", "QtWidgets.pyd"):
        if not (pyside / required).exists():
            raise CommandFailed(f"pruning Qt removed {required}, which the app imports")
    log(f"pruned {removed} unused Qt modules, plugins and tools")


def copy_sources(root: Path) -> None:
    log("copying the application")
    src = root / "src"
    src.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for item in SOURCES:
        path = REPO / item
        if path.is_dir():
            shutil.copytree(path, src / item, ignore=ignore)
        else:
            shutil.copy2(path, src / item)


def compile_bytecode(root: Path, os_name: str) -> None:
    """Precompile everything with hash-checked-free pycs: the launchers run
    with -B (nothing is written into the tree), and archive formats that round
    mtimes (zip) cannot invalidate them."""
    log("precompiling bytecode")
    py = python_exe(root, os_name)
    # Our sources must compile. Third-party trees can ship .py files that are
    # not Python 3 (templates, py2-only helpers) and never get imported;
    # compileall exits 1 for those, so they are reported, not fatal.
    run([py, "-s", "-E", "-m", "compileall", "-q", "--invalidation-mode", "unchecked-hash", root / "src"])
    # -j 0 fans out over every core, which is what makes this fast. It is also
    # the only step that needs a process pool, and a pool needs a pipe: on a
    # host where pipe creation is blocked (a restricted token, some sandboxes)
    # compileall dies before compiling ANYTHING and still exits 1 -- the same
    # exit code a genuinely uncompilable third-party file produces, so the
    # distinction cannot be made from the status. Retrying serially is the
    # cheap way to tell them apart: if the parallel run did nothing, the serial
    # one does the whole job; if it merely tripped over one bad file, the
    # serial run fails on the same file and costs one extra pass, only on a
    # build that was already reporting a failure.
    cmd = [py, "-s", "-E", "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash",
           root / "python"]
    print("   $", " ".join(str(c) for c in cmd), flush=True)
    code = stream(cmd)
    if code:
        serial = [py, "-s", "-E", "-m", "compileall", "-q", "-j", "1",
                  "--invalidation-mode", "unchecked-hash", root / "python"]
        print("    parallel compileall failed; retrying serially", flush=True)
        print("   $", " ".join(str(c) for c in serial), flush=True)
        if stream(serial):
            print("    (some third-party files do not compile; they are left as source)", flush=True)


def install_ffmpeg(root: Path, os_name: str, cache: Path) -> None:
    log("adding ffmpeg and ffprobe")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    lic = root / "licenses"
    lic.mkdir(exist_ok=True)
    wanted = {"ffmpeg", "ffprobe", "ffmpeg.exe", "ffprobe.exe"}
    for url, digest in FFMPEG[os_name]:
        archive = fetch(url, cache, digest)
        if archive.name.endswith(".tar.xz"):
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    base = member.name.rsplit("/", 1)[-1]
                    if member.isfile() and (base in wanted and "/bin/" in member.name or base == "LICENSE.txt"):
                        data = tar.extractfile(member).read()
                        target = bin_dir / base if base in wanted else lic / "FFmpeg-LICENSE.txt"
                        target.write_bytes(data)
        else:
            with zipfile.ZipFile(archive) as zf:
                for name in zf.namelist():
                    base = name.rsplit("/", 1)[-1]
                    if base in wanted:
                        (bin_dir / base).write_bytes(zf.read(name))
                    elif base == "LICENSE.txt":
                        (lic / "FFmpeg-LICENSE.txt").write_bytes(zf.read(name))
    for tool in ("ffmpeg", "ffprobe"):
        path = bin_dir / (tool + (".exe" if os_name == "win" else ""))
        if not path.exists():
            sys.exit(f"{path.name} missing from the ffmpeg download")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_models(root: Path, cache: Path) -> None:
    """Ship all three PP-OCRv6 sizes in `<root>/models`.

    The wheel already carries `small`; `tiny` and `medium` are fetched here so
    a released build never downloads a model, whatever size a folder is set
    to. `videocr.ocr_engine.bundled_model` finds them beside `bin/`, so they
    are read without touching the per-user cache.
    """
    log("adding the PP-OCRv6 tiny and medium models")
    models = root / "models"
    models.mkdir()
    for url, digest in OCR_MODELS:
        source = fetch(url, cache, digest)
        shutil.copy2(source, models / source.name)


def write_notices(root: Path, os_name: str) -> None:
    ffmpeg_src = "BtbN/FFmpeg-Builds (LGPL build), https://github.com/BtbN/FFmpeg-Builds"
    text = f"""OCR Manager bundles third-party software. Each keeps its own license.

- CPython {PYTHON} (python-build-standalone {PBS_RELEASE}): PSF License,
  python/ (see the LICENSE files the interpreter carries).
- Python packages in python/: each package's license is in its
  *.dist-info directory under site-packages.
- FFmpeg (bin/ffmpeg, bin/ffprobe): from {ffmpeg_src}.
  FFmpeg is licensed under the LGPL v2.1+ (GPL v2+ for GPL builds); its
  source is available at https://ffmpeg.org/download.html and from the
  build project above.
- RapidOCR: Apache License 2.0. The PP-OCR models it carries in
  site-packages/rapidocr/models, and the tiny and medium sizes in models/,
  are published by PaddlePaddle (Apache License 2.0).
- ONNX Runtime and DirectML (onnxruntime-directml): MIT License, with the
  DirectML redistributable under the Microsoft Windows SDK terms.
"""
    (root / "licenses").mkdir(exist_ok=True)
    (root / "licenses" / "THIRD-PARTY.txt").write_text(text, encoding="utf-8")


def write_metadata(root: Path, version: str, os_name: str, arch: str) -> None:
    (root / "bundle.json").write_text(
        json.dumps({"version": version, "os": os_name, "arch": arch}, indent=2) + "\n", encoding="utf-8")


def build_tree(root: Path, version: str, os_name: str, arch: str, cache: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    install_python(root, os_name, arch, cache)
    install_dependencies(root, os_name)
    prune_qt(root, os_name)
    copy_sources(root)
    install_ffmpeg(root, os_name, cache)
    install_models(root, cache)
    write_notices(root, os_name)
    write_metadata(root, version, os_name, arch)
    compile_bytecode(root, os_name)


# -- icons ------------------------------------------------------------------

def make_ico(py: Path, out: Path) -> None:
    run([py, "-s", "-E", "-c",
         "import sys; from PIL import Image; Image.open(sys.argv[1]).save(sys.argv[2], "
         "sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])",
         REPO / "resources" / "app-icon.png", out])


# -- the Windows artifacts ---------------------------------------------------

ISCC_LOCATIONS = (
    r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 7\ISCC.exe",
    r"C:\Program Files (x86)\Inno Setup 7\ISCC.exe",
)


def find_iscc() -> str:
    """ISCC on PATH, else the usual install folders.

    Inno Setup 7 installs to `C:\\Program Files\\Inno Setup 7` -- not the x86
    folder version 6 used, and not on PATH -- so hard-coding version 6's path
    made a machine with only 7 fail with "no such file" after the zip had
    already been written. The installer script itself needs no change: 7 still
    reads the `#ifndef`/`#error` dialect this one uses.
    """
    found = shutil.which("iscc")
    if found:
        return found
    for candidate in ISCC_LOCATIONS:
        if Path(candidate).exists():
            return candidate
    raise CommandFailed(
        "Inno Setup's ISCC.exe was not found: put it on PATH or install it in "
        + " or ".join(f'"{p}"' for p in ISCC_LOCATIONS))


def windows_artifacts(tree: Path, version: str, dist: Path, work: Path) -> list[Path]:
    ico = work / "app.ico"
    make_ico(python_exe(tree, "win"), ico)
    shutil.copy2(ico, tree / "app.ico")

    log("compiling the launcher")
    build = work / "launcher"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    shutil.copy2(PACKAGING / "windows" / "launcher.c", build)
    shutil.copy2(PACKAGING / "windows" / "launcher.rc", build)
    shutil.copy2(ico, build / "app.ico")
    run(["rc", "/nologo", "/fo", "launcher.res", "launcher.rc"], cwd=build)
    run(["cl", "/nologo", "/O2", "/W4", "/DUNICODE", "/D_UNICODE", "launcher.c", "launcher.res",
         "/link", "/SUBSYSTEM:WINDOWS", "user32.lib", "/OUT:launcher.exe"], cwd=build)
    shutil.copy2(build / "launcher.exe", tree / f"{APP_NAME}.exe")
    shutil.copy2(PACKAGING / "windows" / "ocr-manager-cli.cmd", tree / "ocr-manager-cli.cmd")

    stem = f"ocr-manager-v{version}-win"
    archive = dist / f"{stem}.zip"
    log(f"writing {archive.name}")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                zf.write(path, f"{stem}/{path.relative_to(tree).as_posix()}")

    iscc = find_iscc()
    log(f"writing {stem}.exe")
    run([iscc, "/Qp", f"/DAppVersion={version}", f"/DSourceDir={tree}", f"/DOutputDir={dist}",
         f"/DOutputBaseFilename={stem}", f"/DIconFile={ico}",
         PACKAGING / "windows" / "installer.iss"])
    return [archive, dist / f"{stem}.exe"]


def tree_path(work: Path) -> Path:
    """Where the runnable bundle root is after a build."""
    return work / "tree"


# -- running the built bundle -------------------------------------------------

def run_bundle(work: Path, args: list[str]) -> int:
    """Run the built bundle's app with a console interpreter, exactly as the
    launchers do (same env and flags): CI's smoke tests go through this."""
    os_name, _ = host()
    root = tree_path(work)
    if not (root / "bundle.json").exists():
        sys.exit(f"no bundle at {root}; build it first")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["OCR_MANAGER_BUNDLE"] = str(root)
    env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", "")
    cmd = [str(python_exe(root, os_name)), *PY_FLAGS, str(root / "src" / "main.py"), *args]
    print("   $", " ".join(cmd), flush=True)
    code = stream(cmd, env=env)
    if code:
        annotate(f"bundle {' '.join(args)} failed", f"exit code {code}\n" + "\n".join(stream.tail))
    return code


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", default=None, help="defaults to app/version.py")
    parser.add_argument("--dist", type=Path, default=REPO / "dist")
    parser.add_argument("--work", type=Path, default=REPO / "build" / "bundle")
    parser.add_argument("--cache", type=Path, default=REPO / ".build-cache")
    parser.add_argument("--bundle-only", action="store_true", help="build the tree, no artifacts")
    if argv[:1] == ["run"]:
        args, rest = parser.parse_known_args(argv[1:])
        if rest[:1] == ["--"]:
            rest = rest[1:]
        return run_bundle(args.work.resolve(), rest)
    args = parser.parse_args(argv)

    os_name, arch = host()
    version = args.version or app_version()
    work, dist, cache = args.work.resolve(), args.dist.resolve(), args.cache.resolve()
    work.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)

    tree = work / "tree"
    log(f"building OCR Manager {version} for {os_name}/{arch}")
    build_tree(tree, version, os_name, arch, cache)
    if args.bundle_only:
        return 0

    outputs = windows_artifacts(tree, version, dist, work)
    for path in outputs:
        print(f"    {path.name}  {path.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if exc.code not in (None, 0) and not isinstance(exc.code, int):
            annotate("build failed", str(exc.code))
        raise
    except BaseException:
        annotate("build failed", traceback.format_exc()[-6000:])
        raise
