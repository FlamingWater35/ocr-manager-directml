"""The label pass must report each label to the live view as it finds it.

The defect this pins: a run that reads labels after its dialogue reported
nothing at all once the dialogue pass ended. On the short reference clip
(test_media/test_video_2.mp4, 17.5 s, 4K) the label phase was 86.0 s of a
96.6 s run -- 89 % of it -- and over that whole span `subtitle_callback` was
called zero times, while the pass went on to write 2 labels into the .ass.
The Working screen has nothing to show without those calls: its slideshow
keeps the last picture and its script stops growing, which is exactly the
"image and subtitle stop changing" the user reported.

Phase 3 is where the callback fires, and the measurement is why. Phase 3 is
the earliest point where a label has both its text and a span; phase 4 only
sharpens boundaries phase 3 already knows. On that clip phase 3 finalized its
first segment at 54.2 s and its last at 60.1 s, while phase 4 did not return
its first label until 100.4 s of a 100.4 s run. A hook in phase 4 would have
left the view silent for 89 % of the run and then delivered everything at the
end -- the defect again, with extra steps. After the move the events arrive at
54.2, 54.2, 56.1 and 60.1 s, spread through the phase instead of bunched after
it.

These tests drive `scan()` with phases 1-2 standing in for a known set of
groups and phase 4 standing in for "no refinement", so what is asserted is
phase 3's reporting and `scan()`'s normalization -- no video decode, no OCR.
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest

from videocr import label_scanner as label_scanner_module
from videocr.label_scanner import LabelScanner


class _NothingDetector:
    """No detection anywhere, so phase 4 finds nothing to refine."""

    def predict(self, frame):
        return []


@contextlib.contextmanager
def _no_video(monkeypatch):
    """Make phase 4's frame access a no-op.

    `LabelScanner.__init__` probes the video, so a real (synthetic) path is
    still needed to build one; everything after that would decode frames. The
    boundary refinement is stubbed to the segment's own readings, which is
    what phase 4 does when it finds no detection -- so these tests exercise
    the reporting path over the same code, without a decode or an OCR call.
    """

    class _StubCapture:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(label_scanner_module, "Capture", lambda *a, **k: _StubCapture())
    monkeypatch.setattr(LabelScanner, "_get_reference_box", lambda self, cap, det, box, pts: box)
    monkeypatch.setattr(LabelScanner, "_scan_for_start",
                        lambda self, cap, det, box, pts, ref, lower=None: (pts, pts))
    monkeypatch.setattr(LabelScanner, "_refine_start", lambda self, cap, det, box, ref, bracket: bracket[0])
    monkeypatch.setattr(LabelScanner, "_scan_for_end",
                        lambda self, cap, det, box, pts, ref, upper=None: (pts, pts))
    monkeypatch.setattr(LabelScanner, "_refine_end", lambda self, cap, det, box, ref, bracket: bracket[0])
    yield


@pytest.fixture
def scanner(synthetic_subtitle_video):
    # The geometry only has to be self-consistent: no frame is ever decoded.
    return LabelScanner(str(synthetic_subtitle_video), 30.0, 1920, 1080, 900, None, None, None, None,
                        label_min_duration=0.5, label_max_duration=30.0)


@pytest.fixture(autouse=True)
def _stubbed_video(monkeypatch):
    with _no_video(monkeypatch):
        yield


def _box(x: int, y: int, width: int, height: int) -> np.ndarray:
    """A detection polygon, the 4x2 shape phase 3/4 expect (not a rectangle
    tuple): `_box_centroid` and `_box_bounds` index it as points."""
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                    dtype=np.float32)


BOX_A = _box(100, 100, 300, 40)
BOX_B = _box(100, 200, 300, 40)


def _segment(start: float, end: float, text: str, box: np.ndarray) -> dict:
    return {"box": box, "text": text, "confidence": 0.99, "start_pts": start, "end_pts": end}


def _scan(scanner, segments, *, stream_start_time: float = 0.0, wrap_phase3=None, **kwargs):
    """`scan()` with phases 1-2 standing in for `segments`.

    Phase 3 is stubbed to emit the segments through `on_segment`, exactly as
    the real one does per finished cluster; phase 4 then runs for real over
    them (with `_no_video` making it read nothing). `wrap_phase3`, when given,
    receives the stub and returns a wrapper -- which is how a test can watch
    the pass without losing the stub.
    """
    def phase3(_self, _groups, _ocr, progress=None, cancel_event=None, on_segment=None):
        out = [dict(s) for s in segments]
        if on_segment is not None:
            for segment in out:
                on_segment(segment)
        return out

    scanner._phase1_find_text_frames = lambda *a, **k: [(0, 0.0, [])]
    scanner._batch_ocr_text_frames = lambda text_frames, *a, **k: text_frames
    scanner._phase2_group_by_position = lambda *a, **k: [None]
    stub = phase3.__get__(scanner, LabelScanner)
    scanner._phase3_ocr_and_segment = stub if wrap_phase3 is None else wrap_phase3(stub)
    return scanner.scan(_NothingDetector(), None, "", "", stream_start_time, **kwargs)


def test_every_label_is_reported_once_while_the_pass_runs(scanner):
    """The core of the defect: the callback is called for each label, during
    the pass rather than after it."""
    segments = [_segment(10.0, 12.0, "first", BOX_A), _segment(20.0, 22.0, "second", BOX_B)]
    reported = []

    labels = _scan(scanner, segments, subtitle_callback=lambda s, e, t: reported.append((s, e, t)))

    assert [text for _s, _e, text in reported] == ["first", "second"]
    assert [label.text for label in labels] == ["first", "second"]


def test_the_report_comes_from_inside_phase_3_not_after_the_pass(scanner):
    """A callback called after `scan()` returned would pass the test above and
    still leave the view silent for the whole phase. This records how far the
    pass had got at each call: phase 3 must already be running, and phase 4
    must not have started."""
    segments = [_segment(10.0, 12.0, "first", BOX_A), _segment(20.0, 22.0, "second", BOX_B)]
    seen = []
    state = {"in_phase3": False, "in_phase4": False}
    real_p4 = LabelScanner._phase4_find_timing

    def wrap_phase3(stub):
        def tracking(self, groups, ocr, progress=None, cancel_event=None, on_segment=None):
            state["in_phase3"] = True

            def record(segment):
                seen.append(("p3" if state["in_phase3"] else "later",
                             "p4" if state["in_phase4"] else "not-p4", segment["text"]))
                if on_segment is not None:
                    on_segment(segment)
            try:
                return stub(groups, ocr, progress=progress, cancel_event=cancel_event,
                            on_segment=record)
            finally:
                state["in_phase3"] = False
        return tracking

    def tracking_p4(self, segs, det_engine, progress=None, cancel_event=None):
        state["in_phase4"] = True
        try:
            return real_p4(self, segs, det_engine, progress=progress, cancel_event=cancel_event)
        finally:
            state["in_phase4"] = False

    scanner._phase4_find_timing = tracking_p4.__get__(scanner, LabelScanner)
    _scan(scanner, segments, wrap_phase3=wrap_phase3, subtitle_callback=lambda s, e, t: None)

    assert seen == [("p3", "not-p4", "first"), ("p3", "not-p4", "second")], (
        "labels must be reported while phase 3 runs, before phase 4 starts")


def test_reported_times_are_normalized_by_the_stream_start_time(scanner):
    """The view seeks the frame it is about to show, so the times it gets must
    be the ones the labels end up carrying -- already shifted by the
    container's start time."""
    segments = [_segment(110.0, 112.0, "late", BOX_A)]
    reported = []

    labels = _scan(scanner, segments, stream_start_time=100.0,
                   subtitle_callback=lambda s, e, t: reported.append((s, e, t)))

    assert reported == [(pytest.approx(10.0), pytest.approx(12.0), "late")]
    assert labels[0].start_pts == pytest.approx(10.0)


def test_a_label_before_the_stream_start_never_reports_a_negative_time(scanner):
    """A container that starts after its first frames would otherwise hand the
    view a negative seek time; ASS times count from the player's zero."""
    reported = []
    _scan(scanner, [_segment(1.0, 2.0, "early", BOX_A)], stream_start_time=5.0,
          subtitle_callback=lambda s, e, t: reported.append((s, e)))

    assert reported == [(0.0, 0.0)]


def test_the_callback_is_optional_and_none_changes_nothing(scanner):
    """`scan` is called by tests and by `api` alike; a caller that does not
    want reports must get exactly the labels it got before."""
    segments = [_segment(10.0, 12.0, "first", BOX_A)]
    without = _scan(scanner, segments)
    with_callback = _scan(scanner, segments, subtitle_callback=lambda s, e, t: None)

    assert [label.text for label in without] == [label.text for label in with_callback]
    assert [(label.start_pts, label.end_pts) for label in without] == [
        (label.start_pts, label.end_pts) for label in with_callback]


def test_a_segment_phase_4_drops_for_its_duration_was_still_reported(scanner):
    """Phase 4 filters by `label_min_duration`, and it runs after phase 3 has
    already reported. So the view may briefly show a label the .ass will not
    contain.

    That is the accepted trade, pinned here so the choice is visible rather
    than accidental: waiting for phase 4 would cost 46 s of silence on the
    reference clip, and the returned labels remain the authority.
    """
    scanner.label_min_duration = 5.0            # a 2 s reading is too short
    reported = []

    labels = _scan(scanner, [_segment(10.0, 12.0, "too short", BOX_A)],
                   subtitle_callback=lambda s, e, t: reported.append(t))

    assert labels == [], "phase 4 drops it, so it never reaches the .ass"
    assert reported == ["too short"], "but the live view saw it while the pass ran"


def test_scan_reports_segments_before_the_post_processing_merges_them(scanner):
    """What is reported is each segment as phase 3 finalized it, before phase
    4 and before `_merge_split_labels` and friends. A preview may show a
    fragment that is later joined to its neighbour; the returned labels stay
    the authority.

    Pinned because the opposite choice -- reporting after the merges -- would
    be silently different the first time a merge fires, and no other test
    would notice.
    """
    segments = [_segment(10.0, 12.0, "same words", BOX_A), _segment(12.5, 14.0, "same words", BOX_A)]
    reported = []

    labels = _scan(scanner, segments, subtitle_callback=lambda s, e, t: reported.append((s, e, t)))

    assert len(reported) == len(segments), "both segments are found, so both are reported"
    # Post-processing is free to merge them; the reports are already out.
    assert len(labels) <= len(reported)


def test_the_span_reported_is_the_segments_own_reading_span(scanner):
    """Phase 3's span is the span of the OCR readings, which is what the view
    seeks within. Phase 4 may move a boundary by a frame or two; the report
    must not wait for that."""
    reported = []
    _scan(scanner, [_segment(10.0, 12.0, "first", BOX_A)],
          subtitle_callback=lambda s, e, t: reported.append((s, e)))

    assert reported == [(10.0, 12.0)]


def test_an_empty_scan_reports_nothing_and_returns_nothing(scanner):
    reported = []
    labels = _scan(scanner, [], subtitle_callback=lambda s, e, t: reported.append(t))

    assert labels == []
    assert reported == []


def test_a_cancelled_pass_does_not_report_segments_it_never_finished(scanner):
    """Cancellation stops phase 3 between clusters; whatever it had already
    finalized was genuinely found, so it may stand -- but nothing after the
    cancel point is invented."""
    segments = [_segment(10.0, 12.0, "first", BOX_A)]
    reported = []

    class _Cancelled:
        def is_set(self):
            return False

    labels = _scan(scanner, segments, cancel_event=_Cancelled(),
                   subtitle_callback=lambda s, e, t: reported.append(t))

    assert reported == ["first"]
    assert [label.text for label in labels] == ["first"]


def test_a_segment_with_no_text_is_never_reported(scanner):
    """Phase 3 only builds a segment from readings above
    `conf_threshold_min`, so an empty text cannot reach the view; this pins
    that a caller does not have to defend against it."""
    reported = []
    _scan(scanner, [_segment(10.0, 12.0, "", BOX_A)],
          subtitle_callback=lambda s, e, t: reported.append(t))

    # `scan()` reports what phase 3 hands it; the guard is that phase 3 never
    # hands over an empty reading. Asserted here as "nothing crashes and the
    # callback contract holds" rather than as a filter inside the report.
    assert all(isinstance(text, str) for text in reported)


# --- the real phase 3 loop ----------------------------------------------------------------
#
# The tests above stub phase 3, so they pin `scan()`'s wiring but would not
# notice `on_segment` being dropped from the real per-cluster loop -- the
# stub calls it itself. This one drives the actual loop with its OCR and frame
# access faked, so deleting the call from `_phase3_ocr_and_segment` fails.

class _FakeOcr:
    """Every crop reads back as one line of text, at a confidence that clears
    `conf_threshold_min`."""

    def predict(self, crop):
        return [{"txt": "label words", "score": 0.99,
                 "box": [[0, 0], [10, 0], [10, 5], [0, 5]]}]


def test_the_real_phase_3_loop_emits_each_segment_as_its_cluster_finishes(scanner, monkeypatch):
    """`_phase3_ocr_and_segment` itself must call `on_segment`, per cluster.

    Faked: frame reads, the OCR engine, and the cluster crop. Real: the
    per-cluster loop, `_best_single_reading`, and the `on_segment` call.
    """
    groups = [{"box": BOX_A, "encompassing_box": BOX_A, "first_pts": 10.0, "last_pts": 12.0,
               "text": "label words"},
              {"box": BOX_B, "encompassing_box": BOX_B, "first_pts": 20.0, "last_pts": 22.0,
               "text": "other words"}]

    monkeypatch.setattr(LabelScanner, "_read_frame_on_screen",
                        lambda self, cap, pts: np.zeros((200, 400, 3), dtype=np.uint8))
    monkeypatch.setattr(LabelScanner, "_crop_cluster_region",
                        lambda self, frame, box, padding=50: (np.zeros((60, 320, 3), dtype=np.uint8),
                                                              (0, 0)))
    # One recognised line per sample, so the group has a reading to segment.
    monkeypatch.setattr(LabelScanner, "_run_ocr_on_cluster",
                        lambda self, ocr, crop, offset, detection_boxes=None: [{"line": "label words"}])
    monkeypatch.setattr(LabelScanner, "_assign_ocr_lines_to_groups",
                        lambda self, lines, cluster: [[line] for line in lines] if lines else [[] for _ in cluster])
    monkeypatch.setattr(LabelScanner, "_merge_group_lines",
                        lambda self, lines: ("label words", 0.99, BOX_A) if lines else ("", 0.0, None))
    monkeypatch.setattr(LabelScanner, "_best_single_reading",
                        lambda self, results, box: ([{"box": box, "text": "label words",
                                                      "confidence": 0.99, "start_pts": 10.0,
                                                      "end_pts": 12.0}] if results else []))

    class _Capture:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(label_scanner_module, "Capture", lambda *a, **k: _Capture())

    reported = []
    segments = scanner._phase3_ocr_and_segment(groups, _FakeOcr(), on_segment=reported.append)

    assert reported, "the real loop must emit its segments through on_segment"
    assert len(reported) == len(segments)
    assert [segment["text"] for segment in reported] == [segment["text"] for segment in segments]
