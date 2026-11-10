"""Auto-detection against the real-world clips in `test_media/`.

These are the only tests in the suite that use real 4K video, so they carry the
`needs_real_media` marker and skip cleanly when the folder is absent (it is
git-ignored -- see `test_media/README.md`).

What they pin, and why each number is what it is:

- **The crop box on both clips.** `test_media/test_video.mkv` is 4K 10-bit HEVC
  with a hand-written embedded English track; `test_video_2.mp4` is a short 4K
  clip. Before the shape filter (`core/detect/crop.py:_drop_tall_polys`) BOTH
  returned `box=None`: a hand silhouette was unioned into the subtitle band on
  the long clip (`ceiling-exceeded+multiple-positions?`) and shared a frame
  with it on the short one (`ceiling-exceeded`). Disabling the filter is
  verified in `tests/test_detect_crop.py` to reproduce that, so these boxes are
  the filter working, not a coincidence.
- **The brightness threshold on both clips.** Before the Otsu split
  (`core/detect/brightness.py:analytic_seed`) the long clip's seed was 102,
  under `IMPLAUSIBLE_SEED`, so it was reported `coloured-text?` and nothing was
  measured at all. 232 is the measured value on that clip.
- **The short clip's probe count.** Its speech window used to be 3.5 s, which
  `vad.probe_times()` filled with ONE candidate, so the box came from a single
  frame and was flagged `low-agreement`. The minimum window span exists for
  that; `agreed >= LOW_AGREEMENT_HITS` is the property, not a specific count.

The end-to-end check is deliberately a WINDOW, not the whole file: the full
349 s 4K pass takes hours, and cue quality is just as well established over a
contiguous stretch that is known to carry dialogue. It asserts that the
produced Chinese cues line up with the embedded English track's timings -- the
strongest statement available, because the two are independent sources (one is
OCR of the picture, the other is what a human transcribed).

Run with::

    $env:PYTHONPATH=".tmp"; .conda\\python.exe -m pytest tests\\test_real_media.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.detect import brightness as B
from core.detect import crop as C
from core.detect import ocr_view
from core.detect import vad

MEDIA = Path(__file__).resolve().parent.parent / "test_media"
LONG = MEDIA / "test_video.mkv"
SHORT = MEDIA / "test_video_2.mp4"
# The embedded track is extracted on demand rather than checked in (the clips
# are git-ignored, and the extraction is a one-liner -- see test_media/README.md).
# Either place is accepted: next to the clips, or in the scratch folder the
# README's command writes to.
EMBEDDED_CANDIDATES = (MEDIA / "embedded_eng.ass",
                       Path(__file__).resolve().parent.parent / ".tmp" / "embedded_eng.ass")

# Measured on the two clips, to the nearest pixel. The boxes are not tight
# around the glyphs: CROP_VERTICAL_PADDING and the baseline clustering widen
# them, which is intended (a threshold applied to a wider band is harmless;
# clipping a line is not).
LONG_BOX = (576, 1428, 2688, 153)
SHORT_BOX = (576, 1398, 2688, 160)
BOX_TOLERANCE_PX = 12

pytestmark = [
    pytest.mark.needs_real_media,
    pytest.mark.skipif(not (LONG.exists() and SHORT.exists()),
                       reason="the git-ignored test_media/ clips are not present"),
]


@pytest.fixture(scope="module")
def engines():
    """One detection engine and one OCR engine for the whole module: leasing is
    expensive (DirectML session setup) and these tests are read-only."""
    from videocr import engine_registry

    with engine_registry.lease_detection_engine(None, True, "small") as det:
        with engine_registry.lease_ocr_engine("ch", None, None, True, "small") as ocr:
            yield det, ocr


def _near(actual, expected, tolerance=BOX_TOLERANCE_PX):
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


# --------------------------------------------------------------------------
# Crop
# --------------------------------------------------------------------------

def test_the_long_clip_gets_a_crop_box_at_all(engines):
    det, _ocr = engines
    duration, _fps = ocr_view.video_timing(str(LONG))

    result = C.detect_crop(str(LONG), duration, det, [], {}, None)

    # The whole point: before the shape filter this was None.
    assert result.box is not None, f"flagged {result.flagged!r}"
    assert _near(result.box, LONG_BOX), f"got {result.box}"


def test_the_long_clip_crop_box_is_auto_applicable(engines):
    det, _ocr = engines
    duration, _fps = ocr_view.video_timing(str(LONG))

    result = C.detect_crop(str(LONG), duration, det, [], {}, None)

    # The false positives it excluded sat entirely above the box, so they are a
    # record of what happened, not a doubt about the box.
    assert result.auto_applicable, f"flagged {result.flagged!r}"
    assert result.flagged in (None, C.FLAG_SHAPE_EXCLUDED)


def test_the_short_clip_gets_a_crop_box_from_more_than_one_frame(engines):
    det, _ocr = engines
    duration, _fps = ocr_view.video_timing(str(SHORT))

    result = C.detect_crop(str(SHORT), duration, det, [], {}, None)

    assert result.box is not None, f"flagged {result.flagged!r}"
    assert _near(result.box, SHORT_BOX), f"got {result.box}"
    # The probe-window floor exists so the convergence stop is reachable; a
    # single contributing frame is what "low-agreement" reported before it.
    assert result.agreed >= C.LOW_AGREEMENT_HITS, (
        f"only {result.agreed} contributing frame(s) of {result.probes_used} probe(s)")
    assert result.auto_applicable, f"flagged {result.flagged!r}"


# --------------------------------------------------------------------------
# Brightness
# --------------------------------------------------------------------------

def test_the_long_clip_gets_a_real_brightness_threshold(engines):
    det, ocr = engines
    duration, _fps = ocr_view.video_timing(str(LONG))

    crop = C.detect_crop(str(LONG), duration, det, [], {}, None)
    result = B.detect_brightness(str(LONG), crop.box, None, det, ocr, None, None)

    # Before the Otsu split: seed 102 -> "coloured-text?" -> nothing measured.
    assert result.seed >= B.IMPLAUSIBLE_SEED, f"seed {result.seed} was rejected"
    assert B.FLAG_COLOURED_TEXT not in (result.flagged or "")
    assert result.auto_applicable, f"flagged {result.flagged!r}"
    assert result.plateau is not None
    assert 200 <= result.value <= 255, f"value {result.value}"


def test_the_short_clip_gets_a_real_brightness_threshold(engines):
    det, ocr = engines
    duration, _fps = ocr_view.video_timing(str(SHORT))

    crop = C.detect_crop(str(SHORT), duration, det, [], {}, None)
    result = B.detect_brightness(str(SHORT), crop.box, None, det, ocr, None, None)

    assert result.auto_applicable, f"flagged {result.flagged!r}"
    assert result.plateau is not None
    assert 200 <= result.value <= 255, f"value {result.value}"


# --------------------------------------------------------------------------
# Speech probing
# --------------------------------------------------------------------------

def test_the_short_clip_offers_enough_probe_candidates():
    duration, _fps = ocr_view.video_timing(str(SHORT))

    window = C._probe_window(duration)
    times = vad.probe_times(str(SHORT), duration, window_frac=window)

    # The clip is shorter than the minimum span, so the window is the whole
    # file and 2.5 s spacing caps the candidates at ~7 -- that is the ceiling,
    # not a shortfall. What matters is that it is no longer the single
    # candidate a 3.5 s fractional window produced, which is what made the box
    # come from one frame and report "low-agreement".
    assert window == (0.0, 1.0), f"a clip shorter than the floor should use all of itself, got {window}"
    assert len(times) >= C.LOW_AGREEMENT_HITS, f"only {len(times)} candidate(s)"


def test_the_long_clip_keeps_the_fractional_probe_window():
    duration, _fps = ocr_view.video_timing(str(LONG))

    # Nothing about the minimum-span floor may move a long file's window.
    assert C._probe_window(duration) == (C.PROBE_WINDOW_START_FRAC,
                                         C.PROBE_WINDOW_END_FRAC)


# --------------------------------------------------------------------------
# End to end: does the detected settings actually produce the subtitles?
# --------------------------------------------------------------------------

# A contiguous stretch the speech probes already know carries dialogue.
WINDOW = ("0:02:18", "0:02:26")
TIME = re.compile(
    r"^Dialogue:\s*\d+,(\d+):(\d+):(\d+)\.(\d+),(\d+):(\d+):(\d+)\.(\d+),([^,]*),(.*)$")


def _cues(ass_text: str, start_floor: float, end_ceiling: float):
    """[(start, end, text)] of Default-style cues inside the window."""
    cues = []
    for line in ass_text.splitlines():
        match = TIME.match(line)
        if not match:
            continue
        fields = [int(value) for value in match.groups()[:8]]
        start = fields[0] * 3600 + fields[1] * 60 + fields[2] + fields[3] / 100.0
        end = fields[4] * 3600 + fields[5] * 60 + fields[6] + fields[7] / 100.0
        text = re.sub(r"\{[^}]*\}", "", match.group(10)).strip()
        if match.group(9) == "Default" and text and start_floor <= start <= end_ceiling:
            cues.append((start, end, text))
    return cues


@pytest.mark.slow
def test_the_detected_settings_produce_cues_that_match_the_embedded_track(engines):
    from videocr import api

    embedded = next((path for path in EMBEDDED_CANDIDATES if path.exists()), None)
    if embedded is None:
        pytest.skip("the embedded English track has not been extracted "
                    "(see test_media/README.md for the ffmpeg command)")

    det, ocr = engines
    duration, _fps = ocr_view.video_timing(str(LONG))
    crop = C.detect_crop(str(LONG), duration, det, [], {}, None)
    bright = B.detect_brightness(str(LONG), crop.box, None, det, ocr, None, None)

    produced_text = api.get_subtitles(
        str(LONG), lang="ch", conf_threshold=95, sim_threshold=82,
        brightness_threshold=bright.value, similar_image_threshold=0.3,
        frames_to_skip=0, use_gpu=True, tier="small",
        crop_x=crop.box[0], crop_y=crop.box[1],
        crop_width=crop.box[2], crop_height=crop.box[3],
        time_ranges=[WINDOW],
    )
    produced = _cues(produced_text, 0.0, float("inf"))
    assert produced, "the OCR pass produced no cues at all"

    truth = _cues(embedded.read_text(encoding="utf-8-sig"), 0.0, float("inf"))
    start = _clock_seconds(WINDOW[0])
    end = _clock_seconds(WINDOW[1])
    truth = [cue for cue in truth if start <= cue[0] <= end]
    assert truth, "the embedded track has no cues in the window"

    # Coverage: every line the human transcribed is also found by the OCR pass.
    missed = [cue for cue in truth
              if not any(p_start <= (cue[0] + cue[1]) / 2.0 <= p_end
                         for p_start, p_end, _t in produced)]
    assert not missed, f"{len(missed)} of {len(truth)} cue(s) missed: {missed}"

    # Timing: measured at 0.01 s on this window, so 0.25 s is a generous bound
    # that still fails loudly if the PTS handling ever drifts.
    deltas = []
    for truth_start, truth_end, _t in truth:
        covering = [(s, e) for s, e, _x in produced
                    if s <= (truth_start + truth_end) / 2.0 <= e]
        best = min(covering, key=lambda c: abs(c[0] - truth_start))
        deltas.append(max(abs(best[0] - truth_start), abs(best[1] - truth_end)))
    assert max(deltas) <= 0.25, f"worst timing delta {max(deltas):.2f}s"


def _clock_seconds(text: str) -> float:
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    return hours * 3600 + minutes * 60 + seconds
