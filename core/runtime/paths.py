"""Where OCR Manager keeps its own files, and whether it runs from a bundle.

Windows only. The build ships a Windows bundle and nothing else, so there is
no per-OS branch left here: the data dir is `%LOCALAPPDATA%\\ocr-manager` and
`$OCR_MANAGER_DATA_DIR` overrides it (tests and CI use that).

- **Data dir** (`data_dir()`): the per-user directory for the downloaded
  OCR models, the logs, and anything else the app owns outside project
  folders.
- **Models dir** (`models_dir()`): `<data>/models`, where RapidOCR keeps the
  recognition models that are not bundled in the wheel (see
  `videocr/ocr_engine.py`). The bundled PP-OCRv6 pair is read from
  site-packages, which a bundle must not write to.
- **Bundle** (`current_bundle()`): a portable build sets
  `$OCR_MANAGER_BUNDLE=<root>`; `<root>/bundle.json` says which build it is.
  With the variable unset the app runs in developer mode.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME = "ocr-manager"
DATA_DIR_ENV = "OCR_MANAGER_DATA_DIR"
BUNDLE_ENV = "OCR_MANAGER_BUNDLE"
BUNDLE_FILE = "bundle.json"
LOG_FILE = "ocr-manager.log"
HOST_OS = "win"
HOST_ARCH = "x86_64"


def host_os() -> str:
    """"win" -- the only OS this build supports.

    Kept as a function (rather than a constant in the call sites) because
    `bundle.json` records it and the state files compare against it.
    """
    return HOST_OS


def host_arch() -> str:
    """"x86_64" -- the only architecture the Windows bundle is built for."""
    return HOST_ARCH


def supported_host() -> bool:
    """True when this process is the platform the app is built for."""
    return sys.platform.startswith("win")


def data_dir(env: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """The app's per-user data directory (not created here)."""
    env = os.environ if env is None else env
    override = env.get(DATA_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    home = Path.home() if home is None else home
    base = env.get("LOCALAPPDATA", "").strip()
    return (Path(base) if base else home / "AppData" / "Local") / APP_DIR_NAME


def models_dir(data: Path | None = None, env: Mapping[str, str] | None = None) -> Path:
    """Where RapidOCR caches recognition models that are not in the wheel."""
    return (data_dir(env) if data is None else Path(data)) / "models"


def logs_dir(env: Mapping[str, str] | None = None) -> Path:
    return data_dir(env) / "logs"


def log_file(env: Mapping[str, str] | None = None) -> Path:
    return logs_dir(env) / LOG_FILE


def source_root() -> Path:
    """The directory holding main.py, app/, core/ and videocr/ (`<root>/src`
    in a bundle, the checkout in developer mode)."""
    return Path(__file__).resolve().parent.parent.parent


class BundleError(RuntimeError):
    """$OCR_MANAGER_BUNDLE is set but does not name a usable bundle."""


@dataclass(frozen=True)
class Bundle:
    root: Path
    version: str
    os: str
    arch: str


def current_bundle(env: Mapping[str, str] | None = None) -> Bundle | None:
    """The bundle this process runs from, or None in developer mode.

    Raises BundleError when the variable is set but its directory is missing
    or its bundle.json is unreadable: a broken bundle must say so rather
    than quietly behave like a developer checkout."""
    env = os.environ if env is None else env
    raw = env.get(BUNDLE_ENV, "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise BundleError(f"{BUNDLE_ENV}={raw} is not a directory")
    path = root / BUNDLE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BundleError(f"{path} is missing") from None
    except (OSError, ValueError) as exc:
        raise BundleError(f"{path} is unreadable: {exc}") from None
    if not isinstance(data, dict):
        raise BundleError(f"{path} is not a JSON object")
    return Bundle(root=root,
                  version=str(data.get("version", "")),
                  os=str(data.get("os") or host_os()),
                  arch=str(data.get("arch") or host_arch()))
