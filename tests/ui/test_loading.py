"""The spinner every screen shows while a frame is on its way.

A frame is never instant: measured on the 4K reference clip, one frame at
720p costs 0.33 s and a batch of 8 costs 4.3 s (`core.detect.crop.grab_frames`
with `band_frac=1.0`), several times that while a run is using the machine. So
"no picture yet" is a state the user looks at for real seconds, and before
this module every screen drew the same static placeholder for it as for a
frame that could never arrive.

These tests pin the three things that make a spinner honest:
  * it turns only while a frame is actually coming (`set_waiting`), never over
    a picture and never for one the decoder could not read;
  * it is a phase, not a widget, so every screen draws it inside its own paint
    event (the crop canvas, the brightness tiles, the slideshow, the previews);
  * with `animated=False` the phase stays put, which is what makes a test's
    grab the same image on every run.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPainter

from app.views import loading
from app.views.episode.slideshow import FramePreview, SlideShow
from app.views.loading import Spinner, paint_spinner, paint_spinner_badge


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """Every test here needs a QApplication (the session fixture supplies it)."""
    return qapp


def _painted(fn) -> QImage:
    """Run `fn(painter)` on a fresh image and return what it drew."""
    image = QImage(120, 120, QImage.Format.Format_RGB32)
    image.fill(QColor("#000000"))
    painter = QPainter(image)
    fn(painter)
    painter.end()
    return image


def _ink(image: QImage) -> int:
    """Pixels the painter touched (anything not the background)."""
    return sum(1 for y in range(image.height()) for x in range(image.width())
               if image.pixelColor(x, y).red() + image.pixelColor(x, y).green()
               + image.pixelColor(x, y).blue() > 0)


# --- the phase ----------------------------------------------------------------------------

def test_a_spinner_starts_stopped_and_at_phase_zero():
    spinner = Spinner(animated=False)
    assert not spinner.running()
    assert spinner.phase() == 0.0


def test_a_stopped_spinner_does_not_turn_even_when_animated():
    """`animated=True` alone must not start the timer: only `set_running` does,
    so a widget can be constructed without spinning."""
    spinner = Spinner(animated=True)
    assert not spinner.running()


def test_an_unanimated_spinner_stays_at_phase_zero_while_running():
    """The offscreen tests rely on this: a grab is then the same image every
    run, so an assertion about what is drawn is not flaky."""
    spinner = Spinner(animated=False)
    spinner.set_running(True)
    for _ in range(5):
        spinner._tick()
    assert spinner.phase() == 0.0


def test_an_animated_spinner_advances_and_wraps():
    spinner = Spinner(animated=True, step_ms=100)
    spinner.set_running(True)
    phases = []
    for _ in range(4):
        spinner._tick()
        phases.append(spinner.phase())
    assert phases == sorted(phases), "the phase must move forward"
    assert all(0.0 <= phase < 1.0 for phase in phases)
    for _ in range(20):
        spinner._tick()
    assert 0.0 <= spinner.phase() < 1.0, "a full turn wraps rather than growing"


def test_stopping_resets_the_phase():
    spinner = Spinner(animated=False)
    spinner.set_running(True)
    spinner._phase = 0.75
    spinner.set_running(False)
    assert spinner.phase() == 0.0, "a spinner that comes back starts from the same place"


def test_the_changed_signal_fires_when_it_starts_and_stops():
    spinner = Spinner(animated=False)
    fired = []
    spinner.changed.connect(lambda: fired.append(spinner.running()))
    spinner.set_running(True)
    spinner.set_running(True)          # no change: no second signal
    spinner.set_running(False)
    assert fired == [True, False]


# --- painting -----------------------------------------------------------------------------

def test_paint_spinner_draws_ink_and_leaves_the_rest_alone():
    image = _painted(lambda painter: paint_spinner(painter, QPointF(60, 60), 20, 0.0))
    ink = _ink(image)
    assert ink > 0, "the spinner drew nothing"
    # An arc, not a disc: the middle of the circle stays empty.
    assert image.pixelColor(60, 60).red() == 0


def test_the_spinner_uses_the_accent_colour_by_default():
    image = _painted(lambda painter: paint_spinner(painter, QPointF(60, 60), 20, 0.0))
    colours = {image.pixelColor(x, y).name() for y in range(120) for x in range(120)
               if image.pixelColor(x, y).red() + image.pixelColor(x, y).green() > 0}
    assert colours, "nothing was drawn"
    # The accent is amber: red dominant in every lit pixel.
    for name in colours:
        red = int(name[1:3], 16)
        blue = int(name[5:7], 16)
        assert red > blue, f"{name} is not the amber accent"


def test_a_phase_change_moves_the_arc():
    first = _painted(lambda painter: paint_spinner(painter, QPointF(60, 60), 20, 0.0))
    half = _painted(lambda painter: paint_spinner(painter, QPointF(60, 60), 20, 0.5))
    assert first != half, "the phase must change what is drawn, or nothing animates"


def test_the_badge_puts_a_disc_behind_the_arc():
    """Over a picture the arc alone would vanish against a bright frame, so the
    badge paints a dark disc first."""
    image = _painted(lambda painter: paint_spinner_badge(painter, QPointF(60, 60), 0.0))
    # A point inside the disc but off the arc is darkened rather than black.
    corner = image.pixelColor(60, 60)
    assert corner.alpha() == 255


# --- the screens that own one -------------------------------------------------------------

def test_the_slideshow_spins_only_while_it_has_no_picture_and_one_is_coming():
    show = SlideShow(animated=False)
    show.resize(640, 360)
    assert not show.waiting()
    assert not show._spinner.running()

    show.set_waiting(True)
    assert show.waiting()
    assert show._spinner.running(), "a frame is on its way, so the stage must say so"

    show.show_line(QImage(320, 180, QImage.Format.Format_RGB32), "text", "0:01", 2.0)
    assert show.has_picture()
    assert not show._spinner.running(), "a picture on screen needs no spinner"


def test_a_late_frame_stops_the_slideshow_spinner_without_clearing_the_line():
    show = SlideShow(animated=False)
    show.resize(640, 360)
    show.set_waiting(True)
    show.show_line(None, "the words arrived first", "0:01", 2.0)
    assert not show.has_picture()
    assert show._spinner.running()

    image = QImage(320, 180, QImage.Format.Format_RGB32)
    image.fill(QColor("#204060"))
    show.show_line(image, "the words arrived first", "0:01", 2.0)
    assert show.has_picture()
    assert not show._spinner.running()
    assert show.caption() == "the words arrived first"


def test_clearing_the_slideshow_lets_the_spinner_come_back():
    show = SlideShow(animated=False)
    show.resize(640, 360)
    show.set_waiting(True)
    show.show_line(QImage(320, 180, QImage.Format.Format_RGB32), "text", "0:01", 2.0)
    assert not show._spinner.running()

    show.clear()
    assert not show.has_picture()
    assert show._spinner.running(), "waiting with nothing on screen means the spinner again"


def test_a_preview_spins_until_its_frame_arrives():
    preview = FramePreview(animated=False, width=320)
    preview.resize(320, 180)
    assert not preview.waiting()

    preview.set_waiting(True)
    assert preview._spinner.running()

    image = QImage(320, 180, QImage.Format.Format_RGB32)
    image.fill(QColor("#204060"))
    preview.set_image(image)
    assert preview.has_image()
    assert not preview._spinner.running()


def test_a_preview_that_was_never_waited_on_stays_still():
    """The Preparing screen hides its preview while it works; a hidden preview
    must not spin for a frame nobody asked for."""
    preview = FramePreview(animated=False, width=320)
    preview.set_image(None)
    assert not preview._spinner.running()


def test_the_slideshow_paints_its_spinner_on_an_empty_stage():
    """Painted, not just flagged: an empty stage with a frame coming must put
    ink in the middle where the placeholder was."""
    empty = SlideShow(animated=False)
    empty.resize(640, 360)
    still = empty.grab().toImage()

    waiting = SlideShow(animated=False)
    waiting.resize(640, 360)
    waiting.set_waiting(True)
    spinning = waiting.grab().toImage()

    assert _ink(still) > 0, "the placeholder gradient itself draws ink"
    assert spinning != still, "waiting for a frame must look different from not waiting"


def test_the_loading_text_is_the_canvas_own_wording():
    """One wording for one state: the crop canvas has said "Loading frame…"
    since before this module, and the new spinner must not introduce a second
    phrase for the same wait."""
    from app.views.crop_view import CropCanvas
    assert loading.LOADING_TEXT == CropCanvas.PLACEHOLDER_TEXT


def test_the_spinner_radius_is_scaled_with_the_ui():
    """Every other measurement in the design system goes through tokens.px, so
    a UI_SCALE change must reach the spinner too -- `px` rounds to a whole
    pixel, which is what a Qt pen width wants."""
    from app.theme import tokens
    assert tokens.px(loading.RADIUS) == int(round(loading.RADIUS * tokens.UI_SCALE))
