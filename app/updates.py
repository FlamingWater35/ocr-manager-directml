"""Is there a newer release than this build?

One request, one comparison:

    GET https://github.com/<owner>/<repo>/releases/latest

GitHub answers that URL with a 302 whose `Location` IS the answer. Measured
on 2026-11-14 from this machine:

    github.com/psf/requests/releases/latest           -> 302 .../releases/tag/v2.34.2
    github.com/<this repo>/releases/latest            -> 302 .../releases

So a repository with a release redirects to its tag page, and one without
any redirects to the plain releases list. The tag in that Location is the
version to compare against `app.version.__version__`, and the Location
itself is the page the window's button opens -- which is why the check
costs one redirect and a few hundred bytes rather than parsing a page.

`api.github.com/repos/.../releases/latest` was the obvious alternative and
is the wrong one: it is rate limited to 60 requests an hour per address and
answered HTTP 403 from this machine within minutes of a test loop, while the
HTML redirect above never did. A desktop app that asks on every launch must
not spend a budget shared with the user's browser and CI.

Nothing here blocks the window. `UpdateChecker.start()` runs the request on
a daemon thread and hands the answer back with a Qt signal, which Qt queues
onto the GUI thread; `check_for_update()` itself is plain Python, so the
tests drive it without a QApplication. A failure of any kind -- offline, DNS,
TLS, a captive portal answering 200, a tag this parser does not understand --
is logged and otherwise silent: an update check must never be the reason the
app looks broken, and it must never put a dialog in front of the work.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from app.version import __version__

logger = logging.getLogger(__name__)

REPOSITORY = "FlamingWater35/ocr-manager-directml"
RELEASES_LATEST = f"https://github.com/{REPOSITORY}/releases/latest"
TAG_MARKER = "/releases/tag/"
# Short on purpose: the answer is one redirect, and a slow network is not a
# reason to keep a socket open. A check that misses its window is simply not
# retried -- the next launch asks again.
TIMEOUT_SECONDS = 4.0
USER_AGENT = f"OCR-Manager/{__version__} (+https://github.com/{REPOSITORY})"
# `OCR_MANAGER_UPDATE_CHECK=0` turns the check off. tests/ui/conftest.py sets
# it for the whole suite: hundreds of windows are constructed there, and a
# suite must not send a request to GitHub per window (nor wait for a timeout
# on a machine that is offline).
ENV_NAME = "OCR_MANAGER_UPDATE_CHECK"
FALSE_WORDS = ("0", "false", "no", "off")


@dataclass(frozen=True)
class Update:
    """A release newer than this build: its version, its tag and its page."""

    version: str        # "3.0.2" -- the tag without its leading v, for display
    tag: str            # "v3.0.2" -- as GitHub names it
    url: str            # the release page the button opens


def update_check_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """False only when `OCR_MANAGER_UPDATE_CHECK` is set to a false word.

    Unset means on: an update check that needs opt-in is one nobody gets.
    """
    value = (os.environ if environ is None else environ).get(ENV_NAME, "")
    return value.strip().lower() not in FALSE_WORDS


def version_key(text: str) -> tuple[tuple[int, ...], int] | None:
    """`"v3.0.2"` -> `((3, 0, 2), 1)`; `"3.0.2-rc1"` -> `((3, 0, 2), 0)`.

    None when `text` holds no leading number at all (`"nightly"`). The
    second element ranks a final release above its own pre-releases, which
    is what makes `3.0.2-rc1` older than `3.0.2` without a semver library --
    `packaging` is a declared dependency, but the whole comparison is a
    dozen lines and a missing module here would cost the user the feature
    rather than an error message.

    A `+` suffix is build metadata and does not lower the rank (`3.0.2+abc`
    is still 3.0.2); any other suffix is a pre-release.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text[:1] in ("v", "V"):
        text = text[1:]
    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", text)
    if match is None:
        return None
    numbers = tuple(int(part) for part in match.group(1).split("."))
    suffix = match.group(2).strip()
    final = 0 if suffix and not suffix.startswith("+") else 1
    return numbers, final


def is_newer(candidate: str, current: str) -> bool:
    """True when `candidate` is a later release than `current`.

    Either side being unparsable answers False: a tag this app cannot read is
    not evidence of an update, and guessing would nag the user about a build
    they already have.
    """
    left, right = version_key(candidate), version_key(current)
    if left is None or right is None:
        return False
    width = max(len(left[0]), len(right[0]))
    # "3.1" and "3.1.0" are the same release, so the short one is padded.
    left_parts = left[0] + (0,) * (width - len(left[0]))
    right_parts = right[0] + (0,) * (width - len(right[0]))
    return (left_parts, left[1]) > (right_parts, right[1])


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow the redirect: its Location is the answer, and
    following it would download the release page for nothing."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):     # noqa: D102 - urllib hook
        return None


def latest_location(timeout: float = TIMEOUT_SECONDS) -> str | None:
    """The `Location` of the releases/latest redirect, or None.

    None covers every "no answer": a network failure, a timeout, or a 200
    (no redirect at all, which is not what GitHub does today but is what a
    proxy or a captive portal does).
    """
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(RELEASES_LATEST, method="GET",
                                     headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:               # 302 with _NoRedirect; 403/404 from a proxy
        if exc.code in (301, 302, 303, 307, 308):
            return exc.headers.get("Location")
        logger.debug("update check: %s answered HTTP %s", RELEASES_LATEST, exc.code)
        return None
    except (urllib.error.URLError, OSError) as exc:     # offline, DNS, TLS, socket timeout
        logger.debug("update check could not reach GitHub: %s", exc)
        return None
    else:
        response.close()
        logger.debug("update check: %s answered %s with no redirect", RELEASES_LATEST, response.status)
        return None


def release_from_location(location: str | None) -> tuple[str, str] | None:
    """(tag, release page) from a redirect target; None when it names no tag.

    GitHub sends `.../releases` while a repository has no release at all, and
    that is not an update -- it is the absence of any version to compare.
    """
    if not location or TAG_MARKER not in location:
        return None
    url = location.split("?", 1)[0].split("#", 1)[0]
    if not url.startswith(("https://github.com/", "http://github.com/")):
        return None
    tag = url.split(TAG_MARKER, 1)[1]
    if not tag or "/" in tag:
        return None
    return tag, url


def check_for_update(current: str = __version__, timeout: float = TIMEOUT_SECONDS,
                     fetch: Callable[[float], str | None] = latest_location) -> Update | None:
    """The newer release, or None when this build is the latest (or unknown).

    `fetch` is the network step, injectable so the tests never leave the
    machine -- and so a caller can reuse the comparison over a different
    source without duplicating it.
    """
    location = fetch(timeout)
    found = release_from_location(location)
    if found is None:
        logger.debug("update check: no release to compare against (%s)", location)
        return None
    tag, url = found
    if not is_newer(tag, current):
        logger.debug("update check: latest release %s is not newer than %s", tag, current)
        return None
    return Update(version=tag[1:] if tag[:1] in ("v", "V") else tag, tag=tag, url=url)


class UpdateChecker(QObject):
    """Runs one `check_for_update()` off the GUI thread and reports the answer.

    Signals, not return values: the check takes as long as a redirect takes,
    and the window must be usable meanwhile. `available` carries the version
    and the release page; `finished` fires either way, once, so a caller (and
    a test) can tell "nothing new" from "still asking".
    """

    available = Signal(str, str)        # version, release page
    finished = Signal()

    def __init__(self, parent: QObject | None = None, *,
                 check: Callable[[], Update | None] | None = None,
                 enabled: bool | None = None):
        super().__init__(parent)
        self._check = check if check is not None else check_for_update
        self._enabled = update_check_enabled() if enabled is None else bool(enabled)
        self._started = False
        self._thread: threading.Thread | None = None

    def enabled(self) -> bool:
        return self._enabled

    def started(self) -> bool:
        """True once `start()` was called (the thread may still be running)."""
        return self._started

    def start(self) -> None:
        """Ask once. Later calls are no-ops: one launch, one request."""
        if not self._enabled or self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, name="update-check", daemon=True)
        self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Join the worker (tests and a clean shutdown); True when it is gone."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        try:
            update = self._check()
        except Exception:                               # noqa: BLE001 - never raise at the user
            logger.debug("update check failed", exc_info=True)
            update = None
        try:
            if update is not None:
                logger.info("a newer release is available: %s (%s)", update.version, update.url)
                self.available.emit(update.version, update.url)
            self.finished.emit()
        except RuntimeError:                            # the window went away while we asked
            logger.debug("update check finished after its window closed")
