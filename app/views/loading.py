"""The spinner the app draws while a frame is on its way.

A frame is never instant. The crop canvas, the Preparing and Done previews
and the Working screen's slideshow all wait on a decode that costs a seek
plus a decode a time -- measured at 0.33 s a frame for a 4K episode, and
several times that while a run is using the machine -- so "no picture yet"
is a state the user spends real seconds looking at. Before this module the
canvas said "Loading frame…" but the episode screens drew the same gradient
placeholder a settled, picture-less screen would: a slow decode and a
broken one looked exactly alike.

`Spinner` is the phase a painted spinner turns through; `paint_spinner`
draws it. Neither is a QWidget: every screen here paints its own frame, so
a spinner widget would have to sit on top of a picture the screen draws and
fight it for the same pixels. An arc drawn last, inside the same paint
event, costs nothing and cannot get out of step with the picture it is
about.

With `animated=False` the phase stays at 0 and no timer runs: a still arc
still says "something is coming", and a test's grab is then the same image
on every run.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen

from app.theme import tokens

SPIN_MS = 1100             # one turn
SPIN_STEP_MS = 60          # how often the phase moves: ~18 fps, plenty for an arc
ARC_SPAN = 100.0           # degrees the lit tail covers
ARC_SEGMENTS = 10          # the tail is this many fading arcs
ARC_WIDTH = 2.2            # mockup px
RADIUS = 11                # mockup px, the spinner's own radius
BADGE_PAD = 4              # mockup px between a badge's spinner and its disc edge
LOADING_TEXT = "Loading frame…"


class Spinner(QObject):
    """A looping 0..1 phase, and the repaint it asks for.

    `changed` fires once a step, and once when the spinner is started or
    stopped, so the widget that owns it can simply repaint: the phase is
    read at paint time, never stored in the picture.
    """

    changed = Signal()

    def __init__(self, parent: QObject | None = None, *, animated: bool = True,
                 step_ms: int = SPIN_STEP_MS):
        super().__init__(parent)
        self._animated = bool(animated)
        self._running = False
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, int(step_ms)))
        self._timer.timeout.connect(self._tick)

    def running(self) -> bool:
        return self._running

    def phase(self) -> float:
        return self._phase

    def set_running(self, on: bool) -> None:
        """Start or stop turning. Stopping resets the phase, so a spinner
        that comes back later starts from the same place."""
        on = bool(on)
        if on == self._running:
            return
        self._running = on
        if not on:
            self._timer.stop()
            self._phase = 0.0
        elif self._animated:
            self._timer.start()
        self.changed.emit()

    def _tick(self) -> None:
        # A still spinner must stay still even if something ticks it: the
        # module's promise is that `animated=False` gives a test the same
        # image on every run, and a tick from a stray timer would break it.
        if not self._animated:
            return
        self._phase = (self._phase + self._timer.interval() / SPIN_MS) % 1.0
        self.changed.emit()


def paint_spinner(painter: QPainter, centre: QPointF, radius: float, phase: float, *,
                  colour: str | QColor | None = None, width: float | None = None) -> None:
    """Draw the spinner centred on `centre`: a lit arc with a tail fading
    behind it, turned `phase` (0..1) of the way round."""
    base = QColor(colour if colour is not None else tokens.ACC)
    pen = QPen(base)
    pen.setWidthF(width if width is not None else max(1.5, tokens.px(ARC_WIDTH)))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    box = QRectF(centre.x() - radius, centre.y() - radius, radius * 2, radius * 2)
    step = ARC_SPAN / ARC_SEGMENTS
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for index in range(ARC_SEGMENTS):
        fade = (index + 1) / ARC_SEGMENTS
        tint = QColor(base)
        tint.setAlphaF(0.10 + 0.90 * fade * fade)
        pen.setColor(tint)
        painter.setPen(pen)
        start = phase * 360.0 + index * step
        painter.drawArc(box, int(start * 16), int(step * 16))
    painter.restore()


def paint_spinner_badge(painter: QPainter, centre: QPointF, phase: float, *,
                        radius: float | None = None) -> None:
    """The spinner over a dark disc, for a screen that already shows a
    picture: it has to stay legible over any frame without covering one."""
    radius = radius if radius is not None else tokens.px(RADIUS)
    disc = radius + tokens.px(BADGE_PAD)
    red, green, blue, _alpha = tokens.TAG_BG
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(red, green, blue, 200))
    painter.drawEllipse(centre, disc, disc)
    painter.restore()
    paint_spinner(painter, centre, radius, phase)
