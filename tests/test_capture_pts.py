"""The decoder's timestamp contract.

Every PTS this app writes into a .ass file comes from here, so the bar is
exact: a frame's PTS must equal what ffprobe reports for that frame, and the
"player's zero" the subtitle times are measured from must be the container's
start, not the first video frame's.

The decoder reads timestamps off ffmpeg itself (-copyts plus the showinfo
filter's pts_time), so these tests compare against ffprobe rather than against
an ordinal computed from a frame rate -- an ordinal would pass even when the
two disagreed.
"""
import json
import subprocess

import cv2
import pytest

from videocr.ffmpeg_capture import Capture, FFmpegCapture


def ffprobe_pts(path, count):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-read_intervals", f"%+#{count}",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    frames = json.loads(out.stdout)["frames"]
    return [float(f["pts_time"]) for f in frames]


def ffprobe_format_start_time(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["start_time"])


def test_pts_matches_ffprobe(synthetic_video):
    expected = ffprobe_pts(synthetic_video, 10)
    got = []
    with Capture(str(synthetic_video)) as cap:
        for _ in range(10):
            ok, _frame = cap.read()
            assert ok
            got.append(cap.get_last_pts())
    assert got == pytest.approx(expected, abs=1e-6)


def test_container_start_time_is_reported(offset_video):
    """The decoder must report a genuine non-zero start, not a hardcoded
    0.0. offset_video's first frame, like its container, starts at ~1.5 s
    (-output_ts_offset), so a backend that always returns 0.0 fails this
    test."""
    expected_start = ffprobe_format_start_time(offset_video)
    assert expected_start == pytest.approx(1.5, abs=1e-3)
    assert ffprobe_pts(offset_video, 1)[0] == pytest.approx(expected_start, abs=1e-3)

    with FFmpegCapture(str(offset_video)) as cap:
        assert cap.get_stream_start_time() == pytest.approx(expected_start, abs=1e-3)


def test_start_time_is_the_container_not_the_first_video_frame(delayed_video_subtitle_clip):
    """Subtitle times count from the player's zero, so a frame's time in the
    output is its PTS minus get_stream_start_time(). The player's zero is the
    container start (libavformat's format start_time, the earliest of all the
    streams) -- what mpv rebases to with --rebase-start-time=yes, its default.

    In this clip the audio, and with it the container, starts at 0 while the
    first frame starts at 0.080 s -- the layout of Jinwu Guard episodes 07,
    08, 09, 12 and 15. Counting from the first video frame instead puts the
    bar's first frame out at 1.00 s, two frames before the 1.080 s the bar is
    actually painted at, which is the 80 ms the line then shows early.
    Rendering this very clip through mpv confirms it: with 1.00 the line is on
    screen two frames before the bar, with 1.08 they appear together.
    """
    with FFmpegCapture(str(delayed_video_subtitle_clip)) as c:
        assert c.get_stream_start_time() == pytest.approx(0.0, abs=1e-9)
        bar_pts = None
        for _ in range(40):
            ok, frame = c.read()
            assert ok
            if frame[315, 320].min() > 200:
                bar_pts = c.get_last_pts()
                break
        assert bar_pts is not None
        assert bar_pts - c.get_stream_start_time() == pytest.approx(1.08, abs=1e-3)


def test_pts_includes_container_start_time(offset_video):
    """The container start-time offset must reach the reported PTS values,
    not just get_stream_start_time(). Read the first frame of offset_video
    and confirm its PTS matches ffprobe's true (offset) pts_time -- this
    fails if the offset is computed but never added in read()."""
    expected = ffprobe_pts(offset_video, 1)
    with FFmpegCapture(str(offset_video)) as c:
        ok, _frame = c.read()
        assert ok
        assert c.get_last_pts() == pytest.approx(expected[0], abs=1e-3)


def test_reseek_to_same_frame_is_not_a_noop(synthetic_video):
    """set() must compare the seek target against the absolute position, not
    against the frames-read-since-last-seek counter.

    Seek to frame 2, read twice (so the relative "frames since seek" counter
    reaches 2 -- the same number as the frame we're about to re-seek to), then
    seek back to frame 2 again. A guard that compares the target against the
    relative counter sees 2 == 2 and wrongly treats this as a no-op, leaving
    the reader positioned past frame 2 instead of seeking back to it.
    """
    expected = ffprobe_pts(synthetic_video, 5)
    with FFmpegCapture(str(synthetic_video)) as c:
        c.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, _frame = c.read()
        assert ok
        assert c.get_last_pts() == pytest.approx(expected[2], abs=1e-3)
        ok, _frame = c.read()
        assert ok
        assert c.get_last_pts() == pytest.approx(expected[3], abs=1e-3)
        assert c.get(cv2.CAP_PROP_POS_FRAMES) == 4

        # Seek back to frame 2. The relative "since last seek" counter is
        # also 2 at this point, which is what makes the stale guard fail.
        c.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, _frame = c.read()
        assert ok

        assert c.get_last_pts() == pytest.approx(expected[2], abs=1e-3)


def test_seek_to_pts_lands_on_the_first_frame_at_or_after(synthetic_video):
    """seek_to_pts() is how the OCR pass jumps to a subtitle's start: the next
    read() must return the first frame at or after the requested time, never
    before it (a frame before would show the previous subtitle) and never two
    frames later (that would clip the subtitle's opening)."""
    expected = ffprobe_pts(synthetic_video, 40)
    with FFmpegCapture(str(synthetic_video)) as c:
        for target in (0.08, 0.1, 0.2, 0.32):
            assert c.seek_to_pts(target) is True
            ok, _frame = c.read()
            assert ok
            assert c.get_last_pts() == pytest.approx(
                next(t for t in expected if t >= target - 1e-6), abs=1e-6)


def test_seek_past_the_end_reports_no_frame(synthetic_video):
    """A seek past the last frame must say so rather than hand back the last
    frame again: the OCR pass uses the False to stop scanning."""
    with FFmpegCapture(str(synthetic_video)) as c:
        assert c.seek_to_pts(60.0) is False
        assert c.read() == (False, None)


def test_grab_advances_exactly_like_read(synthetic_video):
    """grab() skips a frame's conversion, not the frame: interleaving the two
    must still walk the file one frame per call, with identical PTS."""
    expected = ffprobe_pts(synthetic_video, 6)
    got = []
    with FFmpegCapture(str(synthetic_video)) as c:
        ok, _frame = c.read()
        assert ok
        got.append(c.get_last_pts())
        assert c.grab() is True
        got.append(c.get_last_pts())
        ok, _frame = c.read()
        assert ok
        got.append(c.get_last_pts())
        assert c.grab() is True
        got.append(c.get_last_pts())
    assert got == pytest.approx(expected[:4], abs=1e-6)
