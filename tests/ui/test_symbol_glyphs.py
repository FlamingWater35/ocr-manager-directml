"""Every symbol the UI draws must come out in ONE ink colour.

The defect this pins: three of the sixteen symbols the UI puts in front of
words -- the pause bars, the settings gear and the warning triangle -- were
resolved from Windows' colour emoji font, so they drew a blue rounded square,
a purple gear and an orange triangle beside thirteen monochrome glyphs that
took the button's own text colour. The screenshot that prompted this showed
exactly that mixture.

The fix is one family in `tokens.FONT_STACK`: "Segoe UI Symbol" carries all
three in outline form, and Qt reaches it before the emoji font. These tests
pin the property that matters -- the symbols are drawn in the pen colour --
rather than the family name, so a different monochrome face would pass.

`_is_monochrome` draws a glyph with a pure red pen and looks for pixels where
green or blue dominates. A monochrome glyph cannot produce those; a colour
font ignores the pen and does.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter

from app.theme import tokens

# Every symbol a button or a status line puts in front of a word, plus the
# ones drawn on their own. Keep this list in step with the inventory test
# below, which reads the source rather than a hand-written list.
SYMBOLS = ["▶", "⏸", "■", "⤓", "⚙", "✓", "✕", "◆", "▸", "▾", "⤢", "↻", "⇄", "⚠", "◀", "▲"]

# The three that used to come from the emoji font. Named so a failure says
# which regression came back rather than "a symbol".
EMOJI_PRONE = ("⏸", "⚙", "⚠")


def _ink(symbol: str, size: float) -> tuple[int, int]:
    """(ink pixels, foreign-coloured pixels) for `symbol` at `size` points."""
    cell = QImage(96, 96, QImage.Format.Format_RGB32)
    cell.fill(QColor("#000000"))
    painter = QPainter(cell)
    font = QFont()
    font.setFamilies(list(tokens.FONT_STACK))
    font.setPointSizeF(size)
    painter.setFont(font)
    painter.setPen(QColor("#ff0000"))               # pure red: a colour font cannot honour it
    painter.drawText(cell.rect(), int(Qt.AlignmentFlag.AlignCenter), symbol)
    painter.end()
    foreign = ink = 0
    for y in range(cell.height()):
        for x in range(cell.width()):
            pixel = cell.pixelColor(x, y)
            if pixel.red() + pixel.green() + pixel.blue() == 0:
                continue
            ink += 1
            if pixel.green() > 8 or pixel.blue() > 8:
                foreign += 1
    return ink, foreign


@pytest.mark.parametrize("symbol", SYMBOLS, ids=[f"U+{ord(s):04X}" for s in SYMBOLS])
def test_every_symbol_is_drawn_in_the_pen_colour(qapp, symbol):
    ink, foreign = _ink(symbol, tokens.font_pt(tokens.FONT_SIZE_BODY))
    assert ink > 0, f"{symbol} drew nothing at all: no font in the stack has it"
    assert foreign == 0, (
        f"{symbol} was drawn by a colour font ({foreign} of {ink} ink pixels are not red), "
        "so it ignores the button's text colour. Add a monochrome symbol family to "
        "tokens.FONT_STACK, or use a glyph that has an outline form.")


@pytest.mark.parametrize("symbol", EMOJI_PRONE, ids=[f"U+{ord(s):04X}" for s in EMOJI_PRONE])
def test_the_three_that_used_to_be_emoji_stay_monochrome_at_every_size(qapp, symbol):
    """The three are the ones Windows prefers to draw as colour, and a size
    change is enough to lose the outline form if the family ever drops out of
    the stack -- so they are checked at every size the UI actually uses."""
    for size in (tokens.FONT_SIZE_XS, tokens.FONT_SIZE_SCOPE, tokens.FONT_SIZE_SM,
                 tokens.FONT_SIZE_BTN_SM, tokens.FONT_SIZE_BODY, tokens.FONT_SIZE_MD):
        ink, foreign = _ink(symbol, size)
        assert ink > 0 and foreign == 0, f"{symbol} went colour at {size:.2f}pt"


def test_the_symbol_family_sits_before_the_generic_fallback():
    """`sans-serif` is a last resort: a real family after it is never reached."""
    stack = list(tokens.FONT_STACK)
    assert "Segoe UI Symbol" in stack
    assert stack.index("Segoe UI Symbol") < stack.index("sans-serif")


def test_the_stack_keeps_its_cjk_coverage(qapp):
    """The symbol family must not displace the CJK face for Chinese text.

    "Segoe UI Symbol" covers symbols only, so it cannot take a Han glyph from
    the family ahead of it -- but a stack edit is exactly the kind of change
    that could, and Chinese subtitle text is the app's whole subject.
    """
    font = QFont()
    font.setFamilies(list(tokens.FONT_STACK))
    font.setPointSizeF(tokens.font_pt(tokens.FONT_SIZE_BODY))
    ink, foreign = _ink("你好世界", tokens.font_pt(tokens.FONT_SIZE_BODY))
    assert ink > 0, "the stack lost its CJK coverage"
    assert foreign == 0
