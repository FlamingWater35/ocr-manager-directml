"""Where the app's log records go: one file, always, plus the console.

`core.runtime.logfile` sends raw stdout/stderr (print(), ffmpeg's own chatter,
a C library's message) to `<data>/logs/ocr-manager.log`. That is a byte
stream, not logging: every `logger.debug()` in `app/`, `core/` and `videocr/`
went nowhere at all until this module existed, because nothing called
`logging.basicConfig` or added a handler.

`configure()` attaches two handlers to the root logger:

- a `FileHandler` on `<data>/logs/ocr-manager.log`, at DEBUG. The level is
  deliberately the most verbose one: a user reporting a bug sends this file,
  and a record that was never written cannot be recovered. `logfile` already
  keeps the previous run's file as `ocr-manager.log.1` when it takes the
  stream over; `roll_if_large()` covers the other case, a run with a real
  console whose log file is still the previous run's and has grown past
  `LOG_MAX_BYTES`. It only ever rotates a file nothing holds open: on Windows
  a rename fails while a handle is open, which is why `boot()` redirects
  first and configures logging second.
- a `StreamHandler` on stderr at the same level, but only when stderr is a
  real console. Under pythonw.exe there is no console and `logfile` points
  fd 2 at the same file, so a console handler there would write every record
  into the log twice.

The level can be raised for a quieter run with `$OCR_MANAGER_LOG_LEVEL`
(`debug`, `info`, `warning`, `error`, `critical`); an unreadable value is
ignored rather than raising, because this runs before the window exists.

Third-party loggers are not muted: RapidOCR's own records (which model file
it loaded, which provider it chose) are exactly what a decoder bug report
needs, and they arrive here with the same format.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from core.runtime.paths import log_file

LOG_LEVEL_ENV = "OCR_MANAGER_LOG_LEVEL"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES = 8 * 1024 * 1024

# The attribute that marks a handler as ours, so `configure()` can be called
# again (a second `boot()`, a test that re-enters main) without stacking a
# second copy of every record on top of the first. It carries the kind
# ("file"/"console") rather than leaving `describe()` to isinstance() against
# logging's handler classes, which a caller may have replaced.
_HANDLER_MARK = "_ocr_manager_handler"

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def level_from_env(env=None) -> int:
    """The level `$OCR_MANAGER_LOG_LEVEL` asks for, DEBUG when unset."""
    env = os.environ if env is None else env
    raw = (env.get(LOG_LEVEL_ENV) or "").strip().lower()
    return _LEVELS.get(raw, logging.DEBUG)


def level_name(level: int) -> str:
    """`logging.DEBUG` -> "debug", for reporting the level in `--self-test`."""
    return logging.getLevelName(level).lower()


def roll_if_large(path: Path, max_bytes: int = LOG_MAX_BYTES) -> bool:
    """Rename `path` to `path.1` when it is already past `max_bytes`.

    Only useful for a file nothing has opened yet: on Windows a rename fails
    while a handle is open, and `logfile.redirect_to` keeps one for the whole
    session. `boot()` therefore redirects the streams first and calls this
    second, where the file it looks at is either absent (fd 1/2 created it
    already) or the previous run's, closed.
    """
    try:
        if path.stat().st_size <= max_bytes:
            return False
    except OSError:
        return False
    try:
        os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        return False
    return True


def _formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def _ours(logger: logging.Logger):
    return [h for h in logger.handlers if getattr(h, _HANDLER_MARK, False)]


def _drop_ours(logger: logging.Logger) -> None:
    for handler in _ours(logger):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:                                   # noqa: BLE001 - closing must not raise
            pass


def configure(path: Path | None = None, *, level: int | None = None,
              console: bool = True) -> Path | None:
    """Send logging to `path` (default `<data>/logs/ocr-manager.log`) and,
    when `console`, to stderr as well. Returns the file used, or None when
    the file could not be opened.

    Never raises: a log file that cannot be opened (a read-only profile, a
    full disk) must not stop the app from starting. The console handler is
    still installed in that case, so the run is not silent.
    """
    level = level_from_env() if level is None else level
    root = logging.getLogger()
    _drop_ours(root)
    root.setLevel(level)
    formatter = _formatter()

    written = None
    target = log_file() if path is None else Path(path)
    handler = None
    rolled = False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        rolled = roll_if_large(target)
        handler = logging.FileHandler(str(target), mode="a", encoding="utf-8", errors="backslashreplace",
                                      delay=True)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARK, "file")
        root.addHandler(handler)
        # The handler is delayed, so the file is opened by the first record --
        # which is this one. Emitting inside the try keeps the promise that a
        # log file we cannot write never stops the app: without it the open
        # happens later, in whichever thread logs first, and raises there.
        logging.getLogger(__name__).debug("Logging at %s to %s%s%s", level_name(level), target,
                                          " (console too)" if console else " (file only)",
                                          " after rolling the old one to .1" if rolled else "")
        written = target
    except OSError:
        if handler is not None:
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:                               # noqa: BLE001 - closing must not raise
                pass
        written = None

    if console:
        stream = logging.StreamHandler()
        stream.setLevel(level)
        stream.setFormatter(formatter)
        setattr(stream, _HANDLER_MARK, "console")
        root.addHandler(stream)
    return written


def is_configured() -> bool:
    """True when `configure()` has installed its handlers on the root logger."""
    return bool(_ours(logging.getLogger()))


def describe() -> dict:
    """What `--self-test` reports about logging."""
    handlers = _ours(logging.getLogger())
    files = [getattr(h, "baseFilename", None) for h in handlers
             if getattr(h, _HANDLER_MARK, None) == "file"]
    return {
        "level": level_name(logging.getLogger().level),
        "file": files[0] if files else None,
        "console": any(getattr(h, _HANDLER_MARK, None) == "console" for h in handlers),
    }
