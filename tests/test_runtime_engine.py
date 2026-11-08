"""core.runtime: the data dir, bundle detection, and the log file.

There is no engine installer any more: RapidOCR and onnxruntime-directml are
part of the app, so the runtime package only answers "where does this user's
data live" and "is this a portable bundle".
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.runtime import models_dir, prepare_data_dir
from core.runtime.paths import (
    BUNDLE_ENV,
    DATA_DIR_ENV,
    BundleError,
    current_bundle,
    data_dir,
    host_arch,
    host_os,
    log_file,
    logs_dir,
    supported_host,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def test_data_dir_is_localappdata(tmp_path):
    home = tmp_path / "home"
    assert data_dir({"LOCALAPPDATA": "C:/Users/u/AppData/Local"}, home) == \
        Path("C:/Users/u/AppData/Local/ocr-manager")
    assert data_dir({}, home) == home / "AppData" / "Local" / "ocr-manager"
    assert data_dir({DATA_DIR_ENV: str(tmp_path / "d")}, home) == tmp_path / "d"


def test_the_override_wins_over_localappdata(tmp_path):
    env = {DATA_DIR_ENV: str(tmp_path / "override"), "LOCALAPPDATA": "C:/x"}
    assert data_dir(env, tmp_path) == tmp_path / "override"


def test_models_dir_sits_inside_the_data_dir(tmp_path):
    assert models_dir(tmp_path) == tmp_path / "models"
    assert models_dir(env={DATA_DIR_ENV: str(tmp_path / "d")}) == tmp_path / "d" / "models"


def test_logs_go_next_to_the_models(tmp_path):
    assert logs_dir({DATA_DIR_ENV: str(tmp_path)}) == tmp_path / "logs"
    assert log_file({DATA_DIR_ENV: str(tmp_path)}) == tmp_path / "logs" / "ocr-manager.log"


def test_the_host_is_windows_x86_64():
    assert host_os() == "win"
    assert host_arch() == "x86_64"
    assert supported_host() is sys.platform.startswith("win")


def test_prepare_data_dir_creates_the_model_cache(tmp_path):
    data = prepare_data_dir(tmp_path / "data")
    assert data == tmp_path / "data"
    assert (data / "models").is_dir()
    assert prepare_data_dir(tmp_path / "data") == data            # idempotent


def test_current_bundle(tmp_path):
    assert current_bundle({}) is None
    (tmp_path / "bundle.json").write_text(json.dumps({"version": "1.0.0", "os": "win", "arch": "x86_64"}))
    bundle = current_bundle({BUNDLE_ENV: str(tmp_path)})
    assert (bundle.root, bundle.version, bundle.os, bundle.arch) == (tmp_path, "1.0.0", "win", "x86_64")
    assert not hasattr(bundle, "constraints")                    # nothing to constrain any more
    with pytest.raises(BundleError, match="not a directory"):
        current_bundle({BUNDLE_ENV: str(tmp_path / "missing")})
    (tmp_path / "bundle.json").write_text("{nope")
    with pytest.raises(BundleError, match="unreadable"):
        current_bundle({BUNDLE_ENV: str(tmp_path)})
    (tmp_path / "bundle.json").unlink()
    with pytest.raises(BundleError, match="missing"):
        current_bundle({BUNDLE_ENV: str(tmp_path)})


def test_runtime_imports_no_qt_or_onnxruntime():
    code = ("import sys, core.runtime, core.runtime.logfile; "
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('PyQt6', 'PyQt5', 'PySide6', 'onnxruntime', 'rapidocr')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
