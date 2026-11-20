"""No view may put a stray window on the desktop while it is being built.

A widget added to a `QLayout` that is not installed in a widget yet has no
parent, so it is a window: showing it then paints a frame on the desktop.
That is exactly what a launch used to do -- `ScriptPanel.__init__` showed its
"Following" pill while the row holding it was still a free `QHBoxLayout`,
which cost a 656x519 native window for about 15 ms, seen as a small window
flashing up before the real one.

Checking the finished widget is not enough: once the layout is installed the
pill has a parent and the evidence is gone. These tests therefore watch the
moment of the show, which is the only time the defect exists.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from app.views.episode import DoneView, PrepareView, ScriptPanel, WorkingView
from app.views.episode.script import ScriptPanel as ScriptPanelDirect


class _ShowSpy:
    """Records every widget shown while it is a parentless window.

    `isWindow()` is what makes Qt give a widget a native frame, and a widget
    with no parent that is shown gets one -- so this catches the defect at the
    instant it happens rather than after the layout has been tidied up."""

    def __init__(self) -> None:
        self.strays: list[str] = []
        self._real = None

    def __enter__(self) -> _ShowSpy:
        self._real = QWidget.setVisible

        def set_visible(widget, visible):
            if visible and widget.isWindow() and widget.parent() is None:
                self.strays.append(f"{type(widget).__name__}({widget.objectName() or '-'})")
            return self._real(widget, visible)

        QWidget.setVisible = set_visible
        return self

    def __exit__(self, *exc) -> None:
        QWidget.setVisible = self._real


def test_a_script_panel_shows_no_parentless_window(qapp):
    with _ShowSpy() as spy:
        panel = ScriptPanel(animated=False)
    try:
        assert spy.strays == []
    finally:
        panel.close()


def test_a_live_script_panel_shows_no_parentless_window(qapp):
    """`live=True` is the one that shows the follow pill, which is the widget
    that used to escape: it was shown while `foot` was still a free layout."""
    with _ShowSpy() as spy:
        panel = ScriptPanel(animated=False, live=True)
    try:
        assert panel.follow_button.isVisibleTo(panel)
        assert spy.strays == []
    finally:
        panel.close()


def test_the_window_level_screens_show_no_parentless_window(qapp, fake_runner):
    """Preparing, Working and Done, built the way the main window builds them.

    A `ProjectController` is needed because these views talk to one; the fake
    runner keeps it from starting any job."""
    from app.controller import ProjectController

    controller = ProjectController(fake_runner, save_debounce_ms=10)
    try:
        for factory in (PrepareView, WorkingView, DoneView):
            with _ShowSpy() as spy:
                view = factory(controller)
            try:
                assert spy.strays == [], f"{factory.__name__} showed a parentless window"
            finally:
                view.close()
    finally:
        controller.shutdown(timeout=0.5)


def test_the_follow_pill_has_a_parent_before_it_is_shown(qapp):
    """The ordering the fix encodes: install the layout, then show.

    Pinned separately from the spy tests because those would also pass if the
    pill were simply never shown, while this one requires it to be both
    parented and visible."""
    panel = ScriptPanelDirect(animated=False, live=True)
    try:
        button = panel.follow_button
        assert button.parent() is not None
        assert not button.isWindow()
        assert button.isVisibleTo(panel)
    finally:
        panel.close()
