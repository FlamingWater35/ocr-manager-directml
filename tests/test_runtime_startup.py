"""Startup: stdout/stderr redirection without a console, the bootstrap, the
CLI flags, and the device the OCR engine is built on."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.runtime.paths import log_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def _child(code: str, env: dict | None = None, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                          env=dict(os.environ, **(env or {})), timeout=120, check=False, **kwargs)


# --------------------------------------------------------------------------
# core.runtime.logfile (in child processes: it dup2()s fds 1 and 2)
# --------------------------------------------------------------------------

PYTHONW = """
import os, sys
os.close(1); os.close(2)
sys.stdout = None; sys.stderr = None          # what pythonw.exe gives a program
from core.runtime.logfile import needs_redirect, redirect_if_needed
assert needs_redirect()
path = redirect_if_needed(__import__('pathlib').Path(sys.argv[1]))
print('python-level stdout')
sys.stderr.write('python-level stderr\\n')
os.write(1, b'fd 1\\n'); os.write(2, b'fd 2\\n')
from videocr.utils import suppress_output
with suppress_output():                      # os.dup(1)/os.dup(2) must work now
    os.write(1, b'swallowed\\n')
print('after suppress_output', path is not None)
"""


def test_no_console_output_goes_to_the_log_file(tmp_path):
    log = tmp_path / "logs" / "ocr-manager.log"
    log.parent.mkdir()
    log.write_text("previous run\n")
    done = subprocess.run([sys.executable, "-c", PYTHONW, str(log)], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    text = log.read_text()
    for expected in ("python-level stdout", "python-level stderr", "fd 1", "fd 2", "after suppress_output True"):
        assert expected in text
    assert "swallowed" not in text
    assert (tmp_path / "logs" / "ocr-manager.log.1").read_text() == "previous run\n"


def test_a_devnull_console_is_left_alone_on_windows():
    """Windows stats a pipe and NUL alike, so `discard_devnull` cannot tell a
    CI pipe from nowhere; only pythonw's missing streams count as nowhere.
    A pipe is real output and is never redirected either."""
    code = ("from core.runtime.logfile import needs_redirect; "
            "import sys; sys.exit(10 * needs_redirect() + needs_redirect(discard_devnull=True))")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=60, check=False)
    assert done.returncode == 0
    assert _child(code).returncode == 0                  # pipes are real output


def test_missing_streams_are_redirected():
    code = ("import sys; sys.stdout = None; sys.stderr = None; "
            "from core.runtime.logfile import needs_redirect; sys.exit(0 if needs_redirect() else 1)")
    assert _child(code).returncode == 0


# --------------------------------------------------------------------------
# app.bootstrap
# --------------------------------------------------------------------------

@pytest.fixture
def bundle_env(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "bundle.json").write_text(json.dumps({"version": "1.0.0", "os": "win", "arch": "x86_64"}))
    data = tmp_path / "data"
    monkeypatch.setenv("OCR_MANAGER_BUNDLE", str(root))
    monkeypatch.setenv("OCR_MANAGER_DATA_DIR", str(data))
    return root, data


def test_boot_in_developer_mode_does_nothing(monkeypatch, tmp_path):
    from app.bootstrap import boot

    monkeypatch.delenv("OCR_MANAGER_BUNDLE", raising=False)
    monkeypatch.setenv("OCR_MANAGER_DATA_DIR", str(tmp_path / "data"))
    started = boot()
    assert not started.bundled and started.log_path is None
    assert started.data is not None and (started.data / "models").is_dir()


def test_boot_in_a_bundle_reports_the_bundle_and_prepares_the_data_dir(bundle_env):
    from app.bootstrap import boot

    root, data = bundle_env
    started = boot()
    assert started.bundled and started.bundle.root == root
    assert started.data == data and (data / "models").is_dir()
    # pytest's captured streams are pipes, i.e. real output: nothing is
    # redirected. Where they are missing (pythonw.exe) the log lives here.
    assert started.log_path is None
    assert log_file({"OCR_MANAGER_DATA_DIR": str(data)}) == data / "logs" / "ocr-manager.log"


def test_boot_reports_a_broken_bundle(monkeypatch, tmp_path):
    from app.bootstrap import boot

    monkeypatch.setenv("OCR_MANAGER_BUNDLE", str(tmp_path / "nowhere"))
    started = boot()
    assert not started.bundled and "not a directory" in started.bundle_error


def test_engine_problem_is_none_when_the_engine_loads():
    """The engine ships with the app, so there is nothing to install; a
    machine without DirectML is not a problem either (it runs on the CPU)."""
    from app.engine import engine_problem, engine_summary

    summary = engine_summary()
    assert summary["ok"] is True
    assert summary["device"] in ("dml", "cpu")
    assert engine_problem() is None


def test_engine_problem_explains_a_broken_engine(monkeypatch):
    from app import engine

    monkeypatch.setattr(engine, "engine_summary",
                        lambda: {"ok": False, "error": "DllNotFoundError: DirectML.dll"})
    title, text = engine.engine_problem()
    assert title == "OCR engine unavailable"
    assert "DirectML.dll" in text and "Reinstall" in text


# --------------------------------------------------------------------------
# CLI flags (own processes)
# --------------------------------------------------------------------------

def test_version_flag():
    done = subprocess.run([sys.executable, "main.py", "--version"], cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=120, check=False)
    from app.version import __version__
    assert done.returncode == 0 and done.stdout.strip() == __version__


def test_self_test_flag_reports_json():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env.pop("OCR_MANAGER_BUNDLE", None)
    done = subprocess.run([sys.executable, "-m", "app", "--self-test"], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=300, check=False, env=env)
    report = json.loads(done.stdout)
    assert report["imports"]["failed"] == {} and report["imports"]["count"] > 50
    assert set(report["tools"]) == {"ffmpeg", "ffprobe"}
    assert report["decoder"]["backend"] == "ffmpeg"
    assert report["bundle"] is None and report["engine"]["bundled"] is False
    assert done.returncode == (0 if report["ok"] else 1)


def test_the_engine_install_flags_are_gone():
    """There is no engine to install, so nothing offers to install one."""
    done = subprocess.run([sys.executable, "main.py", "--help"], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=120, check=False)
    assert done.returncode == 0
    assert "--download-models" in done.stdout
    assert "install-engine" not in done.stdout and "setup-engine" not in done.stdout


def test_smoke_line_renders_white_text_on_black(qapp):
    from app.cli import SMOKE_TEXT_CJK, SMOKE_TEXT_LATIN, render_line

    image, text, _family = render_line()
    assert image.shape == (128, 1280, 3) and image.dtype.name == "uint8"
    assert text in (SMOKE_TEXT_CJK, SMOKE_TEXT_LATIN)
    assert image.max() > 200 and (image == 0).mean() > 0.8


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------
# videocr: the device an engine is built on
# --------------------------------------------------------------------------

@pytest.fixture
def fresh_dml_check(monkeypatch):
    from videocr import ocr_engine
    monkeypatch.setattr(ocr_engine, "_DML", None)
    return ocr_engine


def test_resolve_device_follows_directml_availability(fresh_dml_check, monkeypatch):
    monkeypatch.setattr(fresh_dml_check, "dml_available", lambda: True)
    assert fresh_dml_check.resolve_device(True) == "dml"
    assert fresh_dml_check.resolve_device(False) == "cpu"
    monkeypatch.setattr(fresh_dml_check, "dml_available", lambda: False)
    assert fresh_dml_check.resolve_device(True) == "cpu"


def test_dml_availability_is_asked_once(fresh_dml_check, monkeypatch):
    calls = []

    class FakeOrt:
        @staticmethod
        def get_available_providers():
            calls.append(1)
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    assert [fresh_dml_check.dml_available() for _ in range(3)] == [True] * 3
    assert calls == [1]


def test_an_onnxruntime_without_directml_reports_cpu(fresh_dml_check, monkeypatch):
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    assert fresh_dml_check.dml_available() is False


def test_the_directml_flag_follows_the_resolved_device(fresh_dml_check):
    """The flag that decides the provider is set from the device, not from
    what the caller asked for."""
    route = fresh_dml_check.route_for("ch")
    on = fresh_dml_check._params(fresh_dml_check.DML, route, None)
    off = fresh_dml_check._params(fresh_dml_check.CPU, route, None)
    assert on["EngineConfig.onnxruntime.use_dml"] is True
    assert off["EngineConfig.onnxruntime.use_dml"] is False
    assert on["Det.model_type"] is not None and on["Rec.lang_type"] == "ch"


def test_the_bundled_models_are_pinned_so_nothing_is_downloaded(fresh_dml_check):
    """A cache dir alone would re-download the 30 MB pair the wheel carries."""
    params = fresh_dml_check._params(fresh_dml_check.DML, fresh_dml_check.route_for("ch"), "/cache")
    assert params["Det.model_path"].endswith("PP-OCRv6_det_small.onnx")
    assert params["Rec.model_path"].endswith("PP-OCRv6_rec_small.onnx")
    assert params["Cls.model_path"].endswith("ch_ppocr_mobile_v2.0_cls_mobile.onnx")
    assert params["Global.model_root_dir"] == "/cache"
    # A language PP-OCRv6 cannot read resolves its recogniser from the cache.
    cyrillic = fresh_dml_check._params(fresh_dml_check.DML, fresh_dml_check.route_for("ru"), "/cache")
    assert "Rec.model_path" not in cyrillic
    assert cyrillic["Rec.lang_type"] == "cyrillic"


def test_the_smoke_line_falls_back_to_pillow_when_qt_draws_nothing(monkeypatch):
    # Qt's offscreen platform on Windows has no font folder and draws a blank
    # line; the smoke test then draws it with Pillow from a system font file.
    from app import cli

    font = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    if not os.path.exists(font):
        pytest.skip("needs DejaVu Sans")
    monkeypatch.setattr(cli, "FONT_FILES", (("/no/such/font.ttf", True), (font, False)))
    image, text, family = cli._render_with_pillow(640, 96)
    assert image.shape == (96, 640, 3) and image.any()
    assert text == cli.SMOKE_TEXT_LATIN and family == "DejaVuSans.ttf"
