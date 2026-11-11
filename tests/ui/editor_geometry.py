"""Whether an editor's text can actually be painted, measured on the widget.

The bug this exists for: `ButtonSymbols.NoButtons` combined with ANY QSS
`border` on a `QAbstractSpinBox` makes Qt lay the internal `QLineEdit` out
**1 px wide**. `text()` still returns the value and the widget still takes its
normal space, so a screenshot shows an empty box with a thin caret and nothing
in the code looks wrong. Measured on PyQt6 6.10: border + NoButtons leaves the
edit 1x21 px; border + UpDownArrows (with the arrows hidden by a 0-width
`::up-button`) leaves it 148 px wide.

The measurement is the width Qt gave the editor's own child line edit, which is
what the value is painted into -- not the widget's width, which stays correct
either way. `MIN_EDITABLE_TEXT_PX` is a floor no legible value can fit under:
even one digit needs several pixels plus the frame's padding.
"""

from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit

MIN_EDITABLE_TEXT_PX = 8


def text_editor(widget):
    """The widget's own child that paints its text, or None."""
    if isinstance(widget, QComboBox):
        return widget.lineEdit()
    if isinstance(widget, QAbstractSpinBox):
        return widget.findChild(QLineEdit)
    if isinstance(widget, QLineEdit):
        return widget
    return None


def text_width(widget) -> int:
    """Width of the area `widget` paints its value into, or -1 when it has
    none (a toggle, a plain label)."""
    editor = text_editor(widget)
    return -1 if editor is None else editor.width()


def shows_its_text(widget) -> bool:
    """False when the widget has a text editor that is too narrow to paint
    anything in. Widgets without one always pass."""
    width = text_width(widget)
    return width < 0 or width >= MIN_EDITABLE_TEXT_PX
