"""Logging: the file every `logger.debug()` goes to, and what `boot()` wires up.

The app's own log file is the artifact a bug report carries, so these tests
hold three things that were all broken at once before `core.runtime.logsetup`
existed: nothing configured logging at all, a record written before the
stdout/stderr redirect could not reach the file, and a second `boot()` stacked
a second copy of every record on top of the first.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.runtime import logsetup

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def restore_root_logger():
    """Put the root logger back exactly as it was, handlers and level alike:
    `configure()` mutates process-global state that other tests also touch."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:                                # noqa: BLE001
                pass
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


# --------------------------------------------------------------------------
# the level
# --------------------------------------------------------------------------

def test_the_default_level_is_debug():
    """Detailed by default: a record that was never written cannot be
    recovered from a user's report."""
    assert logsetup.level_from_env({}) == logging.DEBUG


@pytest.mark.parametrize("raw, level", [
    ("debug", logging.DEBUG),
    ("DEBUG", logging.DEBUG),
    ("  info  ", logging.INFO),
    ("warning", logging.WARNING),
    ("warn", logging.WARNING),
    ("error", logging.ERROR),
    ("critical", logging.CRITICAL),
])
def test_the_level_can_be_raised_from_the_environment(raw, level):
    assert logsetup.level_from_env({logsetup.LOG_LEVEL_ENV: raw}) == level


@pytest.mark.parametrize("raw", ["", "   ", "verbose", "10", "none"])
def test_an_unreadable_level_falls_back_to_debug(raw):
    """This runs before the window exists: a typo in a shell profile must not
    stop the app from starting, and must not silently mute the log either."""
    assert logsetup.level_from_env({logsetup.LOG_LEVEL_ENV: raw}) == logging.DEBUG


def test_the_level_name_is_the_lowercase_logging_name():
    assert logsetup.level_name(logging.DEBUG) == "debug"
    assert logsetup.level_name(logging.WARNING) == "warning"


# --------------------------------------------------------------------------
# configure()
# --------------------------------------------------------------------------

def test_debug_records_reach_the_file(tmp_path):
    log = tmp_path / "logs" / "ocr-manager.log"
    written = logsetup.configure(log, console=False)
    assert written == log
    logging.getLogger("videocr.ffmpeg_capture").debug("frame at %.3f", 1.5)
    logging.getLogger("core.detect.crop").info("box %r", (1, 2, 3, 4))
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = log.read_text(encoding="utf-8")
    assert "frame at 1.500" in text
    assert "videocr.ffmpeg_capture" in text                 # the logger name, not just the message
    assert "INFO" in text and "DEBUG" in text               # and the level
    assert "box (1, 2, 3, 4)" in text


def test_the_file_is_created_even_when_nothing_is_logged(tmp_path):
    """`delay=True` on the handler means the file only appears on the first
    record; the directory must exist either way so a later write cannot fail."""
    log = tmp_path / "logs" / "ocr-manager.log"
    logsetup.configure(log, console=False)
    assert log.parent.is_dir()


def test_configuring_twice_does_not_duplicate_every_record(tmp_path):
    """`boot()` runs once per process, but a test or a re-entered main can run
    it again; the second configure must replace its handlers, not add to them."""
    log = tmp_path / "logs" / "ocr-manager.log"
    logsetup.configure(log, console=False)
    first = len(logsetup._ours(logging.getLogger()))
    logsetup.configure(log, console=False)
    assert len(logsetup._ours(logging.getLogger())) == first
    logging.getLogger("test").warning("only once")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert log.read_text(encoding="utf-8").count("only once") == 1


def test_the_console_handler_can_be_left_off(tmp_path):
    logsetup.configure(tmp_path / "a.log", console=False)
    assert logsetup.describe()["console"] is False
    logsetup.configure(tmp_path / "b.log", console=True)
    assert logsetup.describe()["console"] is True


def test_describe_reports_the_level_and_the_file(tmp_path):
    log = tmp_path / "logs" / "ocr-manager.log"
    logsetup.configure(log, level=logging.INFO, console=False)
    report = logsetup.describe()
    assert report["level"] == "info"
    assert Path(report["file"]) == log
    assert report["console"] is False


def test_nothing_configured_reports_no_file():
    """Before `configure()` -- a test importing a module directly, or the
    `--self-test` report when logging could not be set up -- the description
    must be empty rather than raising."""
    for handler in logsetup._ours(logging.getLogger()):
        logging.getLogger().removeHandler(handler)
    assert logsetup.is_configured() is False
    assert logsetup.describe()["file"] is None


def test_a_log_file_that_cannot_be_opened_does_not_stop_the_app(tmp_path, monkeypatch):
    """A read-only profile or a full disk must not stop the window opening;
    the console handler still goes in, so the run is not silent."""
    def refuse(*args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(logging, "FileHandler", refuse)
    assert logsetup.configure(tmp_path / "logs" / "a.log", console=True) is None
    assert logsetup.is_configured() is True                  # the console handler
    assert logsetup.describe()["file"] is None


# --------------------------------------------------------------------------
# rolling
# --------------------------------------------------------------------------

def test_a_large_log_is_rolled_before_it_is_opened(tmp_path):
    log = tmp_path / "logs" / "ocr-manager.log"
    log.parent.mkdir(parents=True)
    log.write_text("x" * 2048, encoding="utf-8")
    assert logsetup.roll_if_large(log, max_bytes=1024) is True
    assert not log.exists()
    assert (tmp_path / "logs" / "ocr-manager.log.1").read_text(encoding="utf-8") == "x" * 2048


def test_a_small_log_is_left_alone(tmp_path):
    log = tmp_path / "logs" / "ocr-manager.log"
    log.parent.mkdir(parents=True)
    log.write_text("short", encoding="utf-8")
    assert logsetup.roll_if_large(log, max_bytes=1024) is False
    assert log.read_text(encoding="utf-8") == "short"


def test_rolling_a_file_that_does_not_exist_is_not_an_error(tmp_path):
    assert logsetup.roll_if_large(tmp_path / "nope.log") is False


# --------------------------------------------------------------------------
# boot() wiring
# --------------------------------------------------------------------------

def test_boot_configures_logging_in_developer_mode(monkeypatch, tmp_path):
    from app.bootstrap import boot

    monkeypatch.delenv("OCR_MANAGER_BUNDLE", raising=False)
    monkeypatch.setenv("OCR_MANAGER_DATA_DIR", str(tmp_path / "data"))
    started = boot()
    assert started.log_config == tmp_path / "data" / "logs" / "ocr-manager.log"
    assert logsetup.is_configured()
    assert logsetup.describe()["file"] == str(started.log_config)


def test_boot_reports_where_logging_went_even_with_no_console(monkeypatch, tmp_path):
    """Under pythonw there is no console: the file handler must still be
    configured, and the console one left out so records are not written twice
    into the same file (logfile points fd 2 at it)."""
    from app.bootstrap import boot

    monkeypatch.delenv("OCR_MANAGER_BUNDLE", raising=False)
    monkeypatch.setenv("OCR_MANAGER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("app.bootstrap.needs_redirect", lambda **kwargs: True)
    started = boot()
    assert started.log_config is not None
    assert logsetup.describe()["console"] is False


def test_a_broken_data_dir_does_not_stop_the_bundle_check(monkeypatch, tmp_path):
    """`prepare_data_dir` failing is reported, not raised -- and logging is
    configured before it, so the failure itself can be logged."""
    from app.bootstrap import boot

    monkeypatch.delenv("OCR_MANAGER_BUNDLE", raising=False)
    monkeypatch.setenv("OCR_MANAGER_DATA_DIR", str(tmp_path / "data"))

    def boom(*args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr("app.bootstrap.prepare_data_dir", boom)
    started = boot()
    assert started.data is None
    assert logsetup.is_configured()


# --------------------------------------------------------------------------
# --self-test
# --------------------------------------------------------------------------

def test_the_self_test_report_includes_logging(tmp_path):
    env = dict(os.environ, OCR_MANAGER_DATA_DIR=str(tmp_path / "data"))
    done = subprocess.run([sys.executable, "-m", "app", "--self-test"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=300, env=env, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout)
    assert report["logging"]["level"] == "debug"
    assert report["logging"]["file"].endswith("ocr-manager.log")
    assert report["logging"]["console"] is True              # a pipe is real output
