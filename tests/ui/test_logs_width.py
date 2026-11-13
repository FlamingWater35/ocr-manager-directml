"""The logs window's width belongs to the user, never to a file name.

The defect: a section header is a `QPushButton` showing the log KEY, and a
per-file key is the episode's name. A button sizes itself to its text, so the
longest name in the folder became a floor under the whole window -- the window
opened at its default 875 px and then grew past it to fit the name, and a name
wider than the screen grew it past the screen. The body never did this: it is
a `QPlainTextEdit` with `NoWrap` and its own horizontal scrollbar.

`_LogHeader` reports no minimum width and elides into whatever room the window
gives it, so these tests assert the property that matters: a long key changes
neither the header's minimum width nor the window's.
"""
from __future__ import annotations

from app.controller import ProjectController
from app.views.logs import LogsWindow, default_size

# Long enough that a plain button would want several times the window's width.
LONG = ("Slay the Gods S02E11 - the episode with an unreasonably long name "
        "that nobody would ever type by hand but a release group happily ships "
        "[1080p][HEVC][10bit][DDP5.1][Multi-Subs].mkv")


def _window(qapp, fake_runner, tmp_project, names) -> LogsWindow:
    controller = ProjectController(fake_runner, save_debounce_ms=10)
    controller.open_folder(str(tmp_project(names)))
    controller.append_log("Pipeline", "one line")
    for name in names:
        controller.append_log(name, "one line")
    made = LogsWindow(controller)
    made.show()
    qapp.processEvents()
    return made


def _header_text(header) -> str:
    return header.text()


def test_a_long_file_name_does_not_widen_the_logs_window(qapp, fake_runner, tmp_project):
    short = _window(qapp, fake_runner, tmp_project, ["ep01.mkv"])
    long = _window(qapp, fake_runner, tmp_project, [LONG])
    try:
        assert long.width() == short.width() == default_size()[0]
        # The container's minimum is what the scroll area propagates to the
        # window; the header must contribute nothing to it.
        assert long._container.minimumSizeHint().width() < default_size()[0]
    finally:
        short.close()
        long.close()


def test_the_header_reports_no_minimum_width(qapp, fake_runner, tmp_project):
    made = _window(qapp, fake_runner, tmp_project, [LONG])
    try:
        header = made.section(LONG).header
        assert header.minimumSizeHint().width() == 0
        assert header.sizeHint().width() > default_size()[0], (
            "the fixture is wrong: the header's preferred width should be wide")
    finally:
        made.close()


def test_the_header_elides_and_keeps_the_full_key_in_the_tooltip(qapp, fake_runner, tmp_project):
    made = _window(qapp, fake_runner, tmp_project, [LONG])
    try:
        section = made.section(LONG)
        section.set_expanded(True)
        qapp.processEvents()
        assert section.header.full_text() == "▾ " + LONG
        assert section.header.toolTip() == "▾ " + LONG
        assert _header_text(section.header) != section.header.full_text()
        assert _header_text(section.header).endswith("…")
        # The visible text fits the button's own text area, or the ellipsis
        # itself would be clipped by the stylesheet's padding.
        assert (section.header.fontMetrics().horizontalAdvance(_header_text(section.header))
                <= section.header._text_width() + 1)
    finally:
        made.close()


def test_a_short_key_is_not_elided(qapp, fake_runner, tmp_project):
    """Only a name that does not fit is shortened; "Pipeline" must read whole."""
    made = _window(qapp, fake_runner, tmp_project, ["ep01.mkv"])
    try:
        header = made.section("Pipeline").header
        assert _header_text(header) == header.full_text() == "▸ Pipeline"
    finally:
        made.close()


def test_resizing_narrower_re_elides_the_header(qapp, fake_runner, tmp_project):
    """The window is the user's to shrink; the header follows it down."""
    made = _window(qapp, fake_runner, tmp_project, [LONG])
    try:
        section = made.section(LONG)
        section.set_expanded(True)
        qapp.processEvents()
        wide = _header_text(section.header)
        made.resize(360, made.height())
        qapp.processEvents()
        narrow = _header_text(section.header)
        assert narrow != wide
        assert narrow.endswith("…")
        assert section.header.full_text() == "▾ " + LONG
    finally:
        made.close()


def test_the_header_text_still_toggles_the_section(qapp, fake_runner, tmp_project):
    """Eliding is display only: the header is still the collapse control."""
    made = _window(qapp, fake_runner, tmp_project, [LONG])
    try:
        section = made.section(LONG)
        section.set_expanded(True)
        qapp.processEvents()
        assert section.is_expanded()
        section.header.click()
        qapp.processEvents()
        assert not section.is_expanded()
        assert section.header.full_text() == "▸ " + LONG
        assert not section.body.isVisible()
    finally:
        made.close()


def test_the_preferred_width_does_not_follow_the_elided_text(qapp, fake_runner, tmp_project):
    """`sizeHint` must describe the full key, not the shortened one.

    A hint that followed the elided text would shrink as the text shortened,
    so a layout pass could ask again and get a smaller answer -- the window
    would creep closed instead of open. The preferred width is therefore a
    property of the key, not of the room it currently has.
    """
    made = _window(qapp, fake_runner, tmp_project, [LONG])
    try:
        header = made.section(LONG).header
        narrow = header.sizeHint().width()
        made.resize(360, made.height())
        qapp.processEvents()
        assert header.text().endswith("…")
        assert header.sizeHint().width() == narrow
    finally:
        made.close()


def test_setting_the_key_elides_once_and_settles(qapp, fake_runner, tmp_project):
    """No feedback loop between `setText` and the elide it triggers.

    `_elide` calls `setText`, and `changeEvent` calls `_elide`; if that came
    back round, a header would re-elide on every pass. One call per new key is
    the property, and the hint and text must then be stable.
    """
    made = _window(qapp, fake_runner, tmp_project, ["ep01.mkv"])
    try:
        header = made.section("ep01.mkv").header
        calls = []
        original = type(header)._elide

        def counted(self):
            calls.append(1)
            assert len(calls) < 50, "eliding recursed"
            return original(self)

        type(header)._elide = counted
        try:
            header.set_full_text("▾ " + LONG)
            qapp.processEvents()
        finally:
            type(header)._elide = original
        assert len(calls) == 1
        assert len({header.sizeHint().width() for _ in range(3)}) == 1
    finally:
        made.close()
