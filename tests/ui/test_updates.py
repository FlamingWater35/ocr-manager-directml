"""The update check: one redirect, one comparison, one offer in the top bar.

Three things are pinned here, in the order they happen:

- The comparison. `version_key` / `is_newer` decide whether a tag is a later
  release than `app.version.__version__`, and the cases that used to be the
  bug in hand-rolled version comparisons are named one per test: `3.0.2-rc1`
  is older than `3.0.2`, `3.1` and `3.1.0` are the same release, and a tag
  nobody can parse is not evidence of anything.
- The request. GitHub answers `releases/latest` with a 302 whose `Location`
  is either `.../releases/tag/vX.Y.Z` (there is a release) or `.../releases`
  (there is none). Both answers are tested against recorded `Location`
  strings, plus the failures that must stay silent: a timeout, a 403, a
  portal that answers 200 with no redirect.
- The offer. `UpdateChecker` reports on a daemon thread and `MainWindow`
  turns that into a small button in the top bar that opens the release page.
  The window is driven with an injected checker, so no test here touches the
  network -- `tests/ui/conftest.py` turns the real one off for the whole
  suite through `$OCR_MANAGER_UPDATE_CHECK=0`, and one test below asserts
  that switch is honoured, because a suite that silently started making
  requests would still be green.

`QApplication.processEvents()`, not `QTest.qWait`: a raising slot must go
through the window's excepthook (see app/__main__.py's docstring), and a
background thread's signal is delivered by the event loop, not by a wait.
"""
from __future__ import annotations

import threading
import time
import urllib.error

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from app import updates
from app.main_window import UPDATE_DELAY_MS, MainWindow
from app.updates import Update, UpdateChecker, check_for_update, is_newer, release_from_location, version_key
from app.version import __version__
from app.views.topbar import update_text

RELEASE = "https://github.com/FlamingWater35/ocr-manager-directml/releases/tag/v99.0.0"


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("3.0.2", ((3, 0, 2), 1)),
    ("v3.0.2", ((3, 0, 2), 1)),
    ("V3.0.2", ((3, 0, 2), 1)),
    (" 3.0.2 ", ((3, 0, 2), 1)),
    ("3.0.2-rc1", ((3, 0, 2), 0)),
    ("3.0.2-beta.2", ((3, 0, 2), 0)),
    ("3.0.2+build7", ((3, 0, 2), 1)),        # build metadata does not lower the rank
    ("3.1", ((3, 1), 1)),
    ("nightly", None),
    ("", None),
    (None, None),
])
def test_version_key_reads_a_tag(text, expected):
    assert version_key(text) == expected


@pytest.mark.parametrize("candidate, current, newer", [
    ("v3.0.2", "3.0.1", True),
    ("3.0.2", "3.0.1", True),
    ("v3.1.0", "3.0.9", True),
    ("v4.0.0", "3.99.99", True),
    ("v3.0.1", "3.0.1", False),
    ("v3.0.0", "3.0.1", False),
    ("v2.9.9", "3.0.1", False),
    ("3.1", "3.1.0", False),                 # the same release, written shorter
    ("3.1.0", "3.1", False),
    ("v3.0.2-rc1", "3.0.1", True),           # a pre-release of a later version is still later
    ("v3.0.2-rc1", "3.0.2", False),          # ... but not later than the final release it precedes
    ("v3.0.2", "3.0.2-rc1", True),
    ("nightly", "3.0.1", False),             # unparsable: not evidence of an update
    ("v3.0.2", "nightly", False),
])
def test_is_newer(candidate, current, newer):
    assert is_newer(candidate, current) is newer


def test_the_comparison_never_nags_about_the_running_build():
    """The one case that would be visible to every user of a current build."""
    assert is_newer(f"v{__version__}", __version__) is False


# --------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------

@pytest.mark.parametrize("location, expected", [
    (f"https://github.com/x/y/releases/tag/v3.0.2", ("v3.0.2", "https://github.com/x/y/releases/tag/v3.0.2")),
    (f"https://github.com/x/y/releases/tag/3.0.2?expanded=true",
     ("3.0.2", "https://github.com/x/y/releases/tag/3.0.2")),
    ("https://github.com/x/y/releases", None),           # no release at all: nothing to compare
    ("https://github.com/x/y/releases/latest", None),
    ("https://example.test/releases/tag/v3.0.2", None),  # not a GitHub URL: not our answer
    ("https://github.com/x/y/releases/tag/", None),
    ("https://github.com/x/y/releases/tag/v3/extra", None),
    ("", None),
    (None, None),
])
def test_release_from_location(location, expected):
    assert release_from_location(location) == expected


def test_check_for_update_reports_a_newer_release_and_its_page():
    seen = []

    def fetch(timeout):
        seen.append(timeout)
        return "https://github.com/x/y/releases/tag/v99.0.0"

    update = check_for_update("3.0.1", timeout=1.5, fetch=fetch)
    assert update == Update(version="99.0.0", tag="v99.0.0",
                            url="https://github.com/x/y/releases/tag/v99.0.0")
    assert seen == [1.5]


@pytest.mark.parametrize("location", [
    None,                                                # offline / timeout
    "https://github.com/x/y/releases",                    # no release yet
    "https://github.com/x/y/releases/tag/v3.0.1",          # this very build
    "https://github.com/x/y/releases/tag/v3.0.0",          # older
    "https://github.com/x/y/releases/tag/nightly",         # unparsable
])
def test_check_for_update_is_silent_when_there_is_nothing_to_offer(location):
    assert check_for_update("3.0.1", fetch=lambda timeout: location) is None


def test_latest_location_reads_the_redirect_without_following_it(monkeypatch):
    """A 302 is the answer, not a step: following it would download the
    release page. `_NoRedirect` turns it into an HTTPError, which is what
    this test asserts the code reads the Location from."""
    class Response:
        status = 200
        headers: dict = {}

        def close(self):
            pass

    class Opener:
        def __init__(self, result):
            self.result = result
            self.request = None

        def open(self, request, timeout=None):
            self.request = request
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    captured = {}

    def install(result):
        opener = Opener(result)
        captured["opener"] = opener
        monkeypatch.setattr(updates.urllib.request, "build_opener", lambda *handlers: opener)
        return opener

    redirect = urllib.error.HTTPError(updates.RELEASES_LATEST, 302, "Found",
                                      {"Location": RELEASE}, None)
    opener = install(redirect)
    assert updates.latest_location(timeout=2.0) == RELEASE
    assert opener.request.full_url == updates.RELEASES_LATEST
    assert opener.request.get_header("User-agent") == updates.USER_AGENT

    # A 200 with no redirect (a proxy or a portal): no answer, not an error.
    install(Response())
    assert updates.latest_location() is None

    # 403 from a rate limiter, and an offline socket: both are "no answer".
    for failure in (urllib.error.HTTPError(updates.RELEASES_LATEST, 403, "Forbidden", {}, None),
                    urllib.error.URLError("no route to host"),
                    TimeoutError("timed out")):
        install(failure)
        assert updates.latest_location() is None


def test_the_check_is_on_unless_the_switch_turns_it_off():
    assert updates.update_check_enabled({}) is True
    assert updates.update_check_enabled({"OCR_MANAGER_UPDATE_CHECK": ""}) is True
    assert updates.update_check_enabled({"OCR_MANAGER_UPDATE_CHECK": "1"}) is True
    for off in ("0", "false", "no", "off", " FALSE "):
        assert updates.update_check_enabled({"OCR_MANAGER_UPDATE_CHECK": off}) is False


# --------------------------------------------------------------------------
# The checker (thread -> signal)
# --------------------------------------------------------------------------

def _pump_until(predicate, timeout_ms=5000):
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    QApplication.processEvents()
    return predicate()


def test_the_checker_reports_off_the_gui_thread_and_finishes_once(qapp):
    threads, available, finished = [], [], []
    started = threading.Event()

    def check():
        started.set()
        threads.append(threading.current_thread().name)
        return Update(version="99.0.0", tag="v99.0.0", url=RELEASE)

    checker = UpdateChecker(check=check, enabled=True)
    checker.available.connect(lambda version, url: available.append((version, url)))
    checker.finished.connect(lambda: finished.append(threading.current_thread().name))
    checker.start()
    assert checker.started()
    assert _pump_until(lambda: finished)
    assert started.is_set()
    assert threads == ["update-check"]                    # the request ran off the GUI thread
    assert available == [("99.0.0", RELEASE)]
    assert finished == [threading.current_thread().name]  # ... and the answer arrives on it
    assert checker.wait(1.0)


def test_the_checker_asks_once_and_says_nothing_when_up_to_date(qapp):
    calls, available, finished = [], [], []

    def check():
        calls.append(1)
        return None

    checker = UpdateChecker(check=check, enabled=True)
    checker.available.connect(lambda *args: available.append(args))
    checker.finished.connect(lambda: finished.append(1))
    checker.start()
    checker.start()                                       # one launch, one request
    assert _pump_until(lambda: finished)
    assert calls == [1] and available == [] and len(finished) == 1


def test_a_raising_check_is_logged_not_raised(qapp, caplog):
    finished = []

    def check():
        raise RuntimeError("the network stack exploded")

    checker = UpdateChecker(check=check, enabled=True)
    checker.finished.connect(lambda: finished.append(1))
    with caplog.at_level("DEBUG", logger="app.updates"):
        checker.start()
        assert _pump_until(lambda: finished)
    assert "update check failed" in caplog.text


def test_a_disabled_checker_never_starts_a_thread():
    checker = UpdateChecker(enabled=False)
    checker.start()
    assert not checker.started()
    assert checker.wait(0.1)                              # nothing to join


def test_the_suite_turns_the_real_check_off(qapp, monkeypatch):
    """conftest.py's `_no_update_requests` fixture, verified rather than
    trusted: a window built by a test must not schedule a request."""
    window = MainWindow()
    try:
        assert window.update_checker.enabled() is False
        assert window._update_timer is None
        assert window.topbar.update_button.isHidden()
    finally:
        window._closing = True
        window.close()


# --------------------------------------------------------------------------
# The offer in the top bar
# --------------------------------------------------------------------------

@pytest.fixture
def controller(qapp, fake_runner):
    """A real controller over the fake runner, as in test_main_window.py --
    the window talks to nothing else, and this module drives no jobs."""
    from app.controller import ProjectController

    made = ProjectController(fake_runner, save_debounce_ms=10)
    yield made
    made.shutdown(timeout=0.5)


@pytest.fixture
def make_window(controller):
    """`MainWindow(controller)`, at the size the app opens at.

    Declared here rather than in conftest.py because only the two modules
    that build a window want it; the teardown is not a user, so it never
    answers closeEvent's questions.
    """
    windows = []

    def make(**kwargs) -> MainWindow:
        window = MainWindow(controller, **kwargs)
        window.resize(1440, 900)
        windows.append(window)
        return window

    yield make
    for window in windows:
        window._closing = True
        window.close()
        window.deleteLater()


@pytest.fixture
def opening():
    """The URLs a window opened, instead of a browser."""
    opened = []

    def open_url(url: QUrl):
        opened.append(url.toString())
        return True

    return opened, open_url


def test_an_available_update_becomes_a_button_that_opens_its_page(make_window, opening):
    opened, open_url = opening
    checker = UpdateChecker(check=lambda: Update("99.0.0", "v99.0.0", RELEASE), enabled=True)
    window = make_window(update_checker=checker, open_url=open_url)
    window.show()

    assert window.topbar.update_button.isHidden()          # nothing to offer yet
    checker.start()
    assert _pump_until(lambda: not window.topbar.update_button.isHidden())
    button = window.topbar.update_button
    assert button.text() == "↑ 99.0.0 available"
    assert "99.0.0" in button.toolTip()
    assert window.topbar.update_url() == RELEASE

    button.click()
    assert opened == [RELEASE]


def test_the_update_button_takes_no_room_until_there_is_an_update(make_window, opening):
    """A current build must look exactly as it did before this feature: the
    button is built once, hidden, and `refresh()` -- which runs on every
    project, file and activity signal -- leaves it alone."""
    opened, open_url = opening
    window = make_window(open_url=open_url)
    window.show()
    QApplication.processEvents()
    hidden_width = window.topbar.sizeHint().width()

    window.topbar.show_update("99.0.0", RELEASE)
    QApplication.processEvents()
    assert not window.topbar.update_button.isHidden()
    shown_width = window.topbar.sizeHint().width()
    assert shown_width > hidden_width

    window.topbar.refresh()                               # the bar's own refresh keeps the offer
    QApplication.processEvents()
    assert not window.topbar.update_button.isHidden()
    assert window.topbar.update_button.text() == "↑ 99.0.0 available"

    window.topbar.clear_update()
    QApplication.processEvents()
    assert window.topbar.update_button.isHidden()
    assert window.topbar.sizeHint().width() == hidden_width


def test_the_offer_survives_a_run_and_a_new_folder(make_window, tmp_project, opening):
    """It is read-only, like Logs: a run hides Folder settings and Start but
    not this, and opening another folder does not withdraw it."""
    opened, open_url = opening
    window = make_window(open_url=open_url)
    window.show()
    window.open_folder(str(tmp_project(fixture="slay")))
    window.topbar.show_update("99.0.0", RELEASE)
    QApplication.processEvents()
    assert not window.topbar.update_button.isHidden()

    window.topbar.refresh()
    QApplication.processEvents()
    assert not window.topbar.update_button.isHidden()

    window.open_folder(str(tmp_project(fixture="slay")))
    window.topbar.refresh()
    QApplication.processEvents()
    assert not window.topbar.update_button.isHidden()
    assert window.topbar.update_url() == RELEASE


def test_the_offer_is_a_real_button_with_a_visible_focus_ring(make_window, qapp):
    """It is keyboard reachable: Tab stops on it and the theme draws the ring
    (an ID selector would otherwise outrank `QPushButton:focus`)."""
    from app.theme import qss, tokens

    window = make_window()
    window.topbar.show_update("99.0.0", RELEASE)
    button = window.topbar.update_button
    assert button.focusPolicy() != 0                       # Qt.FocusPolicy.NoFocus
    sheet = qss.build_stylesheet()
    assert "QPushButton#UpdateButton:focus" in sheet
    rule = sheet[sheet.index("QPushButton#UpdateButton {"):]
    assert f"color: {tokens.ACC};" in rule                 # the offer is accent ink ...
    assert f"border: 1px dashed {tokens.ACC_DIM};" in rule  # ... inside a dashed annotation


def test_the_window_checks_for_updates_after_its_delay(make_window, opening):
    """The schedule, end to end: the timer is armed for UPDATE_DELAY_MS and
    firing it runs the checker. Injected checker and an explicit delay, so
    the assertion does not depend on a wall clock."""
    opened, open_url = opening
    checker = UpdateChecker(check=lambda: Update("99.0.0", "v99.0.0", RELEASE), enabled=True)
    window = make_window(update_checker=checker, open_url=open_url)
    window._check_for_updates(delay_ms=0)
    assert window._update_timer is not None and window._update_timer.isActive()
    assert not checker.started()                           # not before the delay is up
    assert _pump_until(checker.started)
    assert _pump_until(lambda: not window.topbar.update_button.isHidden())


def test_the_default_delay_is_the_promised_two_and_a_half_seconds():
    """The brief's "2-3 seconds after the app opens", as one number a later
    edit cannot quietly move out of that window."""
    assert 2000 <= UPDATE_DELAY_MS <= 3000


def test_a_window_that_closes_before_its_check_takes_the_timer_with_it(make_window, opening):
    """Closing in the first seconds is normal (a mis-double-click). The
    pending timer is the window's own child, so it goes with it rather than
    firing into a checker whose window is gone."""
    opened, open_url = opening
    checker = UpdateChecker(check=lambda: Update("99.0.0", "v99.0.0", RELEASE), enabled=True)
    window = make_window(update_checker=checker, open_url=open_url)
    window._check_for_updates(delay_ms=0)
    timer = window._update_timer
    window._closing = True
    window.close()
    assert timer.parent() is window
    assert not timer.isActive()


def test_the_button_text_names_the_version():
    assert update_text("3.0.2") == "↑ 3.0.2 available"
