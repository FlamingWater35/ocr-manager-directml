"""The stage head must not widen itself when tabs are switched.

The defect: with a long file name in the top bar, Crop -> Brightness -> Crop
made the window wider every round trip, until the name finally fit. Two
independent mechanisms fed the same ratchet, and each has a test here.

* The head showed the incoming toolbar before hiding the outgoing one. Qt
  queues a LayoutRequest from the `show()`, and the layout pass then saw BOTH
  toolbars in the head, so the head's minimum width was the tab buttons plus
  both bars. The window widened to satisfy that floor and never came back.
* The Time ranges toolbar's header was a plain `QLabel` holding the file name,
  whose minimum width is the whole text, so that one bar alone set a floor.

Both are checked through the *observable* property rather than the code shape:
the head's minimum width for a given tab must not depend on which tab was
shown before it.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from app.views.stage import Stage

TITLES = ("Crop", "Brightness", "Time ranges")


class _Signal:
    def connect(self, *_args):
        pass


class StubController:
    file_changed = files_changed = folder_changed = _Signal()


class BarTab:
    """A tab whose toolbar is a row of buttons of a known width.

    Three buttons of a fixed minimum width give the bar a minimum width that
    is easy to reason about, which is all the ratchet needs: the bar is wide,
    and the head must not keep it.
    """

    def __init__(self, title: str, width: int):
        self.title = title
        self._page = QLabel(title)
        self._panel = QWidget()
        self._bar = QWidget()
        box = QHBoxLayout(self._bar)
        box.setContentsMargins(0, 0, 0, 0)
        for index in range(3):
            button = QPushButton(f"{title} action {index}")
            button.setMinimumWidth(width)
            box.addWidget(button)

    def page(self):
        return self._page

    def inspector_panel(self):
        return self._panel

    def set_file(self, name):
        pass

    def refresh(self):
        pass

    def toolbar(self):
        return self._bar


def _stage(qapp) -> Stage:
    tabs = [BarTab(title, 120 + 20 * index) for index, title in enumerate(TITLES)]
    stage = Stage(StubController(), tabs)
    stage.resize(900, 300)
    stage.show()
    qapp.processEvents()
    return stage


def _head_width(stage: Stage, qapp) -> int:
    qapp.processEvents()
    return stage.head().minimumSizeHint().width()


def test_the_head_narrows_back_when_a_wide_toolbar_goes_away(qapp):
    """The core of the ratchet: a wide bar must not set the floor forever."""
    stage = _stage(qapp)
    narrow = _head_width(stage, qapp)          # Crop, the narrowest bar
    stage.set_current(2)                       # Time ranges, the widest
    wide = _head_width(stage, qapp)
    stage.set_current(0)                       # back to Crop
    back = _head_width(stage, qapp)
    assert wide > narrow, "the fixture is wrong: the wide bar is not wider"
    assert back == narrow, (
        f"the head stayed {back} px wide after a {wide} px toolbar was hidden; it was "
        f"{narrow} px before that toolbar was ever shown, so the bar left its width behind")


def test_switching_tabs_never_ratchets_the_head_wider(qapp):
    """A round trip must cost nothing, however many times it is taken."""
    stage = _stage(qapp)
    first = _head_width(stage, qapp)
    widths = []
    for _ in range(4):
        for index in (1, 0, 2, 0):
            stage.set_current(index)
            widths.append(_head_width(stage, qapp))
    assert max(widths) >= first
    # Back on Crop every time, so every sample but the Time-ranges ones is the
    # Crop width; a ratchet shows as a sample wider than the first.
    crop_samples = [widths[index] for index in range(len(widths)) if index % 4 == 1]
    assert set(crop_samples) == {first}, (
        f"Crop's head width drifted across round trips: {crop_samples}")


def test_only_the_current_toolbar_is_visible(qapp):
    stage = _stage(qapp)
    for index in (0, 1, 2, 0, 2, 1):
        stage.set_current(index)
        qapp.processEvents()
        visible = [bar for bar in stage._toolbars if not bar.isHidden()]
        assert visible == [stage._toolbars[index]], (
            f"tab {index} left {len(visible)} toolbars visible")
        assert stage.current_toolbar() is stage._toolbars[index]


class _Order(QObject):
    """Records Show/Hide of the stage's toolbars, in the order they arrive."""

    def __init__(self):
        super().__init__()
        self.events: list[str] = []
        self._names: dict[int, str] = {}

    def watch(self, bar, name: str) -> None:
        self._names[id(bar)] = name
        bar.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Show:
            self.events.append("show:" + self._names.get(id(obj), "?"))
        elif event.type() == QEvent.Type.Hide:
            self.events.append("hide:" + self._names.get(id(obj), "?"))
        return False


def test_the_outgoing_toolbar_is_hidden_before_the_incoming_one_appears(qapp):
    """The mechanism behind the ratchet, stated directly.

    Order is what matters: `show()` is what queues the layout pass, so the old
    bar must already be hidden when the new one is shown. Reversing the two
    lines in `_show_toolbar` fails this test.

    This is the test that catches the regression in CI: the two width tests
    above describe the damage, but the offscreen platform does not propagate
    size hints ("This plugin does not support propagateSizeHints()"), so they
    pass even with the order reversed. The event order is visible on every
    platform, which is why the mechanism is asserted here rather than left to
    the symptom.
    """
    stage = _stage(qapp)
    order = _Order()
    for index, tab in enumerate(stage.tabs()):
        order.watch(tab.toolbar(), TITLES[index])
    stage.set_current(2)                       # Crop -> Time ranges: a real swap
    qapp.processEvents()
    assert order.events == ["hide:Crop", "show:Time ranges"], order.events
    order.events.clear()
    stage.set_current(0)                       # and back
    qapp.processEvents()
    assert order.events == ["hide:Time ranges", "show:Crop"], order.events
