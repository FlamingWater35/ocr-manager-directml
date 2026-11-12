"""The stage tab bar's focus and selection, which used to disagree.

The defect: clicking a tab moved the fill to it but left the keyboard focus
(and so the amber focus ring) on the tab it came from, so two tabs read as
active at once. Three separate mechanisms produced the same look, and each
has a test here:

* a click gave a `Qt.FocusPolicy.TabFocus` button no focus at all;
* all three buttons were Tab stops, so Tab walking the bar moved the ring
  while the fill stayed put;
* `[on="true"]` came later in the sheet than `:focus`, so even a focused
  selected tab lost its ring.

The invariant every test below checks is the same one the screenshot asked
for: exactly one tab looks active, and it is the current one.
"""
from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from app.theme import qss, tokens
from app.views.stage import Stage

TITLES = ("Crop", "Brightness", "Time ranges")


class StubTab:
    """The minimum `StageTab`: a page, a panel, and the two no-op hooks."""

    def __init__(self, title: str):
        self.title = title
        self._page = QLabel(title)
        self._panel = QWidget()

    def page(self):
        return self._page

    def inspector_panel(self):
        return self._panel

    def set_file(self, name):
        pass

    def refresh(self):
        pass


class _Signal:
    def connect(self, *_args):
        pass


class StubController:
    file_changed = files_changed = folder_changed = _Signal()

    def project(self):
        return None


def make_stage(qapp) -> Stage:
    stage = Stage(StubController(), [StubTab(title) for title in TITLES])
    stage.resize(900, 300)
    return stage


def _on_buttons(stage: Stage) -> list[bool]:
    return [bool(button.property("on")) for button in stage.tab_buttons()]


def _focused_index(stage: Stage) -> int | None:
    return next((index for index, button in enumerate(stage.tab_buttons())
                 if button.hasFocus()), None)


def _assert_one_active_tab(stage: Stage) -> None:
    """Exactly one tab is filled, and it is the current one."""
    on = _on_buttons(stage)
    assert on.count(True) == 1, on
    assert on.index(True) == stage.current_index(), on


def test_clicking_a_tab_moves_the_focus_with_the_fill(qapp):
    """The defect in one line: the fill moved, the focus ring did not."""
    stage = make_stage(qapp)
    try:
        stage.show()
        qapp.processEvents()
        QTest.mouseClick(stage.tab_buttons()[1], Qt.MouseButton.LeftButton)
        qapp.processEvents()
        assert stage.current_index() == 1
        _assert_one_active_tab(stage)
        assert _focused_index(stage) == 1, "the focus ring must be on the tab that is filled"
    finally:
        stage.deleteLater()


def test_clicking_the_tab_already_showing_still_parks_the_focus_on_it(qapp):
    """A click on the current tab is a no-op for the model but must not leave
    the ring on a neighbour."""
    stage = make_stage(qapp)
    try:
        stage.show()
        qapp.processEvents()
        stage.tab_buttons()[2].setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()
        QTest.mouseClick(stage.tab_buttons()[0], Qt.MouseButton.LeftButton)
        qapp.processEvents()
        assert stage.current_index() == 0
        assert _focused_index(stage) == 0
        _assert_one_active_tab(stage)
    finally:
        stage.deleteLater()


def test_a_programmatic_switch_does_not_steal_the_focus_from_a_page(qapp):
    """The other direction: the model changing under a tab must not yank the
    focus into the tab bar out of whatever the user is typing in.

    Qt itself moves the focus when the page it was on is hidden by the
    stack, so what is asserted here is the part the stage decides: the ring
    must not land on a tab."""
    stage = make_stage(qapp)
    editor = QWidget()
    editor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    layout = QVBoxLayout(stage.current_tab().page())
    layout.addWidget(editor)
    try:
        stage.show()
        qapp.processEvents()
        editor.setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()
        assert qapp.focusWidget() is editor
        stage.set_current(2)
        qapp.processEvents()
        assert stage.current_index() == 2
        assert _focused_index(stage) is None, "a programmatic switch must not move the ring into the bar"
        _assert_one_active_tab(stage)
    finally:
        stage.deleteLater()


def test_the_tab_bar_is_one_tab_stop(qapp):
    """Tab enters the bar on the current tab and leaves again; only the
    current tab accepts Tab focus, or Tab would walk the bar and leave the
    ring on a tab the fill is not on."""
    stage = make_stage(qapp)
    try:
        policies = [button.focusPolicy() for button in stage.tab_buttons()]
        assert policies[0] == Qt.FocusPolicy.TabFocus
        assert policies[1] == policies[2] == Qt.FocusPolicy.ClickFocus
        stage.set_current(2)
        policies = [button.focusPolicy() for button in stage.tab_buttons()]
        assert policies[2] == Qt.FocusPolicy.TabFocus
        assert policies[0] == policies[1] == Qt.FocusPolicy.ClickFocus
    finally:
        stage.deleteLater()


def test_tab_walking_the_window_passes_the_bar_once(qapp):
    """The observable consequence of the policy above: Tab from a widget
    before the stage lands on the current tab, and the next Tab has left the
    bar -- not stepped to its neighbour."""
    window = QWidget()
    outer = QVBoxLayout(window)
    before, after = QLabel("before"), QLabel("after")
    for label in (before, after):
        label.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    stage = Stage(StubController(), [StubTab(title) for title in TITLES])
    outer.addWidget(before)
    outer.addWidget(stage)
    outer.addWidget(after)
    window.resize(900, 500)
    try:
        window.show()
        window.activateWindow()
        qapp.processEvents()
        stage.set_current(1)
        before.setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()

        QTest.keyClick(before, Qt.Key.Key_Tab)
        qapp.processEvents()
        assert _focused_index(stage) == 1, "Tab enters the bar on the current tab"
        _assert_one_active_tab(stage)

        QTest.keyClick(stage.tab_buttons()[1], Qt.Key.Key_Tab)
        qapp.processEvents()
        assert qapp.focusWidget() is after, "the next Tab leaves the bar instead of walking it"
        _assert_one_active_tab(stage)
    finally:
        window.close()
        window.deleteLater()


def test_left_and_right_move_the_selection_not_just_the_focus(qapp):
    """Arrow keys are how the bar is navigated now that Tab does not walk it,
    so they must move the fill (and the page), not only the ring."""
    stage = make_stage(qapp)
    seen: list[int] = []
    stage.tab_changed.connect(seen.append)
    try:
        stage.show()
        qapp.processEvents()
        stage.tab_buttons()[0].setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()
        QTest.keyClick(stage.tab_buttons()[0], Qt.Key.Key_Right)
        qapp.processEvents()
        assert stage.current_index() == 1
        assert stage.page_host().currentIndex() == 1
        _assert_one_active_tab(stage)
        assert _focused_index(stage) == 1
        QTest.keyClick(stage.tab_buttons()[1], Qt.Key.Key_Left)
        qapp.processEvents()
        assert stage.current_index() == 0
        assert seen == [1, 0]
        _assert_one_active_tab(stage)
    finally:
        stage.deleteLater()


def test_the_arrows_wrap_around_the_bar(qapp):
    """Three tabs in a row that wraps: Right on the last is the first, Left
    on the first is the last. Anything else is a dead end at the edge."""
    stage = make_stage(qapp)
    try:
        stage.show()
        qapp.processEvents()
        stage.set_current(2)
        stage.tab_buttons()[2].setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()
        QTest.keyClick(stage.tab_buttons()[2], Qt.Key.Key_Right)
        qapp.processEvents()
        assert stage.current_index() == 0
        QTest.keyClick(stage.tab_buttons()[0], Qt.Key.Key_Left)
        qapp.processEvents()
        assert stage.current_index() == 2
        _assert_one_active_tab(stage)
    finally:
        stage.deleteLater()


def test_other_keys_reach_the_button(qapp):
    """The event filter must not swallow the keys a button already handles:
    Space still presses the focused tab (a no-op here, but it must not be
    consumed by the filter) and Tab must still leave."""
    stage = make_stage(qapp)
    try:
        stage.show()
        qapp.processEvents()
        stage.set_current(1)
        stage.tab_buttons()[1].setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()
        QTest.keyClick(stage.tab_buttons()[1], Qt.Key.Key_Space)
        qapp.processEvents()
        assert stage.current_index() == 1
        assert _focused_index(stage) == 1
        _assert_one_active_tab(stage)
    finally:
        stage.deleteLater()


def test_a_focused_selected_tab_keeps_its_ring():
    """The QSS half: `[on="true"]` is later in the sheet than `:focus`, so
    without a rule that outranks both, a focused selected tab painted exactly
    like an unfocused one -- the ring was invisible in the state a keyboard
    user is in most of the time."""
    sheet = qss.build_stylesheet()
    assert 'QPushButton#StageTab[on="true"]:focus' in sheet
    rule = sheet[sheet.index('QPushButton#StageTab[on="true"]:focus'):]
    declarations = rule[:rule.index("}")]
    assert f"border: 1px solid {tokens.ACC};" in declarations
    # It must come after the plain selected rule it overrides, or the sheet
    # order would put the LINE2 border back on top.
    assert (sheet.index('QPushButton#StageTab[on="true"]:focus')
            > sheet.index('QPushButton#StageTab[on="true"] {'))
    # ... and the fill stays, so the tab still reads as the selected one.
    assert f"border-bottom-color: {tokens.PANEL2};" in declarations
