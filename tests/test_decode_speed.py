"""The two decode-path measurements that made this app 3.9x faster.

Both were found by measuring, and both are the kind of change that quietly
reverts: a future edit that "restores" a few frames of pipe buffering, or that
makes the crop stage fall back to slicing in Python, undoes the speedup without
breaking any output -- so nothing would catch it except a test like this.

1. ``_PIPE_BUFSIZE`` must stay below one frame.  ``io.BufferedReader`` bypasses
   its buffer when ``read(n)`` asks for more than it holds; ask for less and
   every frame is copied twice.  Measured: 24.3 fps against 55.4 fps on the 4K
   clip.  The regression is invisible in the output, so it is pinned here.

2. A crop must never fall back to "no crop stage at all".  Before this, a
   source whose native/output ratio was not a power of two (3840x1640 -> 1080,
   which is most 4K) was refused, so the pipe carried the whole 2528x1080 frame
   and videocr/video.py sliced the band out in numpy: 98% of every frame
   decoded, converted, copied through a pipe and then thrown away.  Measured
   60.7 fps against 65.4 fps on the pipe, and 176 s against 45 s for a real
   30 s OCR window once the whole pipeline is counted.

The exactness of the tail crop is pinned in tests/test_capture_crop.py, which
already compares every crop plan against a numpy slice of the full frame.
"""
import subprocess

import numpy as np
import pytest

from videocr import ffmpeg_capture
from videocr.ffmpeg_capture import FFmpegCapture


def _make_clip(path, width, height, frames=6, rate=25, pix_fmt="yuv420p",
               codec="libx264"):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i",
         f"testsrc2=size={width}x{height}:rate={rate}:duration={frames / rate}",
         "-pix_fmt", pix_fmt, "-c:v", codec, str(path)],
        check=True, capture_output=True,
    )


# --------------------------------------------------------------------------
# 1. the pipe buffer
# --------------------------------------------------------------------------

def test_the_pipe_buffer_is_smaller_than_one_frame(tmp_path):
    """`read()` must stay on the bypass path, or every frame is copied twice.

    The rule is "smaller than one frame", not "equal to N bytes", so this
    checks it against a real decoded frame size.  The clip is 320x240 -- the
    smallest source the app is expected to read -- because a constant that only
    wins at 4K would silently do nothing here: 1 MB is bigger than a 225 KB
    frame, and that is exactly the mistake this test caught when the constant
    was first set.
    """
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240)

    with FFmpegCapture(str(src)) as cap:
        frame_bytes = cap._frame_size()

    assert ffmpeg_capture._PIPE_BUFSIZE < frame_bytes, (
        f"the pipe buffer ({ffmpeg_capture._PIPE_BUFSIZE} B) is not smaller than "
        f"a {frame_bytes} B frame, so every read copies the frame twice "
        "(measured 2.3x slower: see _PIPE_BUFSIZE)"
    )


def test_the_pipe_buffer_is_not_tiny(tmp_path):
    """The other direction: a buffer far below one frame is pointless churn.

    Below the bypass threshold the exact value stops mattering (64 KB and
    4 MB measured the same), so this only guards against someone "fixing" the
    test above by setting the buffer to something absurd like 1 byte.
    """
    assert ffmpeg_capture._PIPE_BUFSIZE >= 64 << 10


def test_a_read_returns_exactly_one_frame(tmp_path):
    """The bypass path must not produce short reads.

    ``bufsize=0`` measured fastest of all and is wrong: an unbuffered pipe
    returns whatever the OS has (32 KB here), which desynchronises the frame
    and PTS pairing.  This pins that a normal read gets a whole frame, which is
    what makes the buffer choice safe rather than merely fast.
    """
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240, frames=6)

    with FFmpegCapture(str(src)) as cap:
        expected = (cap._out_h(), cap._out_w(), 3)
        for _ in range(6):
            ok, frame = cap.read()
            assert ok
            assert frame.shape == expected
            assert frame.nbytes == cap._frame_size()


# --------------------------------------------------------------------------
# 2. the crop always reaches the filter chain
# --------------------------------------------------------------------------

# 3840x1640 -> 1080 is the real clip's geometry: 1640/1080 = 41/27, so the
# native plan is refused and the tail plan has to take over.  This is the
# case that used to decode the whole frame.
_REFUSED_NATIVE = (3840, 1640)
_REFUSED_TARGET = 1080


def test_a_non_power_of_two_ratio_still_crops_in_the_chain(tmp_path):
    """The regression that cost 98% of every decoded frame.

    Before the tail crop, this geometry left `_crop_graph_active` False and
    the pipe carried the full 2528x1080 frame.  Asserting the crop is active
    AND that the pipe is band-sized is what makes the speedup structural: a
    future "simplification" that deletes the tail plan has to delete this test.
    """
    src = tmp_path / "uhd.mp4"
    _make_clip(src, *_REFUSED_NATIVE, frames=4)
    band = (379, 940, 1770, 100)

    with FFmpegCapture(str(src), decode_target_height=_REFUSED_TARGET) as cap:
        full_w, full_h = cap._out_w(), cap._out_h()
        assert (full_w, full_h) == (2528, 1080), (
            f"test geometry drifted: output is {full_w}x{full_h}"
        )

    with FFmpegCapture(str(src), decode_target_height=_REFUSED_TARGET,
                       crop_rect=band) as cap:
        assert cap._crop_graph_active, (
            "the crop was refused entirely, so the pipe carries the full frame"
        )
        assert cap._crop_tail == band, "the tail plan did not take over"
        assert cap._out_w() == band[2] and cap._out_h() == band[3], (
            f"the pipe is {cap._out_w()}x{cap._out_h()}, not the {band[2]}x{band[3]} band"
        )


def test_the_tail_crop_is_exactly_the_slice_it_replaces(tmp_path):
    """The speedup must not come from cropping the wrong pixels.

    This is the same invariant tests/test_capture_crop.py enforces for the
    native plan, restated for the geometry that actually needs the tail plan:
    the cropped frame has to equal the numpy slice of the uncropped frame.
    """
    src = tmp_path / "uhd.mp4"
    _make_clip(src, *_REFUSED_NATIVE, frames=4)
    band = (379, 940, 1770, 100)

    with FFmpegCapture(str(src), decode_target_height=_REFUSED_TARGET) as cap:
        full = [cap.read()[1].copy() for _ in range(4)]
    with FFmpegCapture(str(src), decode_target_height=_REFUSED_TARGET,
                       crop_rect=band) as cap:
        assert cap._crop_tail == band
        cropped = [cap.read()[1].copy() for _ in range(4)]

    x, y, w, h = band
    for index, (reference, got) in enumerate(zip(full, cropped)):
        expected = reference[y:y + h, x:x + w]
        assert got.shape == expected.shape
        assert np.array_equal(got, expected), (
            f"frame {index}: max abs diff "
            f"{int(np.abs(got.astype(int) - expected.astype(int)).max())}"
        )


def test_a_tail_cropped_capture_does_not_ask_to_be_sliced_again():
    """`_crop_slice is None` while `_crop_graph_active` is True.

    The two flags mean different things now, and callers have to test the
    right one: `_crop_slice` says "the chain produced a padded box, slice the
    band out of it", `_crop_graph_active` says "the chain already cropped".
    Testing `_crop_slice` in videocr/video.py or core/detect/ocr_view.py would
    slice the band a second time and read a corner of it.  That was a real bug
    while this change was being made; this test is what keeps the two flags
    from being "simplified" back into one.
    """
    capture = FFmpegCapture.__new__(FFmpegCapture)
    capture._crop_request = None
    capture._width, capture._height = _REFUSED_NATIVE
    capture._output_width, capture._output_height = 2528, _REFUSED_TARGET
    capture._scale_factor = _REFUSED_TARGET / _REFUSED_NATIVE[1]
    capture._crop_native = None
    capture._crop_tail = None
    capture._crop_scaled_size = None
    capture._crop_slice = None
    capture._crop_graph_active = False
    capture.path = "synthetic"

    capture._plan_crop = FFmpegCapture._plan_crop.__get__(capture)
    capture._plan_native_crop = FFmpegCapture._plan_native_crop.__get__(capture)
    capture._plan_tail_crop = FFmpegCapture._plan_tail_crop.__get__(capture)
    capture._crop_request = (379, 940, 1770, 100)
    capture._plan_crop()

    assert capture._crop_graph_active is True
    assert capture._crop_slice is None
    assert capture._crop_native is None


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
def test_an_exact_ratio_still_uses_the_cheap_native_plan(scale):
    """The tail plan is the fallback, not the default.

    A power-of-two ratio (the 1080p and 720p cases, and every test clip in
    tests/test_capture_crop.py) must keep using the native plan, which scales
    only the band.  If the tail plan ever became the only plan, those sources
    would silently start scaling the full frame.
    """
    capture = FFmpegCapture.__new__(FFmpegCapture)
    capture._crop_request = None
    capture._width = 320
    capture._height = int(240 * scale) or 240
    capture._output_height = 240
    capture._output_width = 320
    capture._scale_factor = 1.0
    capture._crop_native = None
    capture._crop_tail = None
    capture._crop_scaled_size = None
    capture._crop_slice = None
    capture._crop_graph_active = False
    capture.path = "synthetic"

    capture._plan_crop = FFmpegCapture._plan_crop.__get__(capture)
    capture._plan_native_crop = FFmpegCapture._plan_native_crop.__get__(capture)
    capture._plan_tail_crop = FFmpegCapture._plan_tail_crop.__get__(capture)
    capture._crop_request = (64, 60, 192, 64)
    capture._plan_crop()

    assert capture._crop_graph_active
    assert capture._crop_native is not None, "an exact ratio should use the native plan"
    assert capture._crop_tail is None


# --------------------------------------------------------------------------
# 3. hardware decode is decided by measurement, not by assumption
# --------------------------------------------------------------------------

def test_hardware_decode_is_off_unless_asked_for(monkeypatch, tmp_path):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240)
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, "1")

    with FFmpegCapture(str(src), use_gpu=False) as cap:
        assert cap.use_gpu is False


def test_hardware_decode_is_off_unless_enabled(monkeypatch, tmp_path):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240)
    monkeypatch.delenv(ffmpeg_capture.HWACCEL_ENV, raising=False)

    with FFmpegCapture(str(src), use_gpu=True) as cap:
        assert cap.use_gpu is False, "hardware decode must stay opt-in"


@pytest.mark.parametrize("value", ["0", "", "no", "off", "maybe"])
def test_the_switch_only_accepts_an_explicit_yes(monkeypatch, value):
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, value)
    assert ffmpeg_capture._hwaccel_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_the_switch_accepts_the_usual_spellings(monkeypatch, value):
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, value)
    assert ffmpeg_capture._hwaccel_enabled() is True


def test_a_pixel_difference_disables_hardware_for_that_file(monkeypatch, tmp_path):
    """The whole point of measuring: a difference must mean software decode.

    The two paths are stubbed rather than relying on a machine having a GPU
    driver that behaves a particular way, so this asserts the DECISION.  The
    real difference is measured in .tmp/verify_bitdepth.py: 8-bit sources
    differ on ~40% of pixels, 10-bit sources are identical.
    """
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240)
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, "1")
    calls = []

    def fake_grab(self, chain, frame_bytes, seek, hardware):
        calls.append(hardware)
        return b"\x00" * frame_bytes if not hardware else b"\x01" * frame_bytes

    monkeypatch.setattr(FFmpegCapture, "_grab_one_frame", fake_grab)

    with FFmpegCapture(str(src), use_gpu=True) as cap:
        assert cap.use_gpu is False
    assert calls == [False, True], "both paths must actually be measured"


def test_an_identical_frame_enables_hardware_for_that_file(monkeypatch, tmp_path):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240)
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, "1")

    def fake_grab(self, chain, frame_bytes, seek, hardware):
        return b"\x2a" * frame_bytes

    monkeypatch.setattr(FFmpegCapture, "_grab_one_frame", fake_grab)

    with FFmpegCapture(str(src), use_gpu=True) as cap:
        assert cap.use_gpu is True


def test_a_failed_hardware_probe_means_software_decode(monkeypatch, tmp_path):
    """No d3d11va, no DX11 device, or a chain that will not start: all False."""
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240)
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, "1")

    def fake_grab(self, chain, frame_bytes, seek, hardware):
        return b"\x00" * frame_bytes if not hardware else None

    monkeypatch.setattr(FFmpegCapture, "_grab_one_frame", fake_grab)

    with FFmpegCapture(str(src), use_gpu=True) as cap:
        assert cap.use_gpu is False


def test_the_hardware_verdict_is_memoised_per_file(monkeypatch, tmp_path):
    """A scan opens a capture per probe; the check must not run each time."""
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240)
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, "1")
    calls = []

    def fake_grab(self, chain, frame_bytes, seek, hardware):
        calls.append(hardware)
        return b"\x2a" * frame_bytes

    monkeypatch.setattr(FFmpegCapture, "_grab_one_frame", fake_grab)

    with FFmpegCapture(str(src), use_gpu=True) as first:
        assert first.use_gpu is True
    with FFmpegCapture(str(src), use_gpu=True) as second:
        assert second.use_gpu is True

    assert calls == [False, True], (
        f"the verdict was re-measured: {len(calls)} probe(s) for two captures"
    )


def test_the_hardware_probe_compares_the_whole_chain(tmp_path, monkeypatch):
    """The probe must decode through the real chain, not just the decoder.

    Comparing the decoder's output alone would miss a difference introduced by
    the tone map, the crop or the scale -- which are exactly the stages that
    turn a different decoder output into different pixels.
    """
    src = tmp_path / "uhd.mp4"
    _make_clip(src, *_REFUSED_NATIVE, frames=4)
    seen = []

    def fake_grab(self, chain, frame_bytes, seek, hardware):
        seen.append((chain, frame_bytes))
        return b"\x2a" * frame_bytes

    monkeypatch.setattr(FFmpegCapture, "_grab_one_frame", fake_grab)
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, "1")

    with FFmpegCapture(str(src), decode_target_height=_REFUSED_TARGET,
                       crop_rect=(379, 940, 1770, 100), use_gpu=True) as cap:
        assert cap.use_gpu is True

    chain, frame_bytes = seen[0]
    assert "crop=1770:100:379:940" in chain, chain
    assert "scale=2528:1080" in chain, chain
    assert frame_bytes == 1770 * 100 * 3


def _probe_seek(tmp_path, monkeypatch, frames, rate=25):
    """The seek the hardware check actually used, for a clip of N frames."""
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240, frames=frames, rate=rate)
    seen = []

    def fake_grab(self, chain, frame_bytes, seek, hardware):
        seen.append(seek)
        return b"\x2a" * frame_bytes

    monkeypatch.setattr(FFmpegCapture, "_grab_one_frame", fake_grab)
    monkeypatch.setenv(ffmpeg_capture.HWACCEL_ENV, "1")
    with FFmpegCapture(str(src), use_gpu=True) as cap:
        assert cap.use_gpu is True
    assert len(seen) == 2, f"both paths must be probed, got {seen}"
    return seen[0]


def test_the_hardware_probe_skips_the_first_frame(tmp_path, monkeypatch):
    """Frame 0 is often black on both paths, so it proves nothing.

    This caught a real no-op: the seek was written as
    ``min(start + 1.0, start)``, which is always ``start``, so every probe
    decoded frame 0 while the comment above it promised otherwise.
    """
    assert _probe_seek(tmp_path, monkeypatch, frames=125) == pytest.approx(1.0)


def test_the_hardware_probe_does_not_seek_past_a_short_clip(tmp_path,
                                                           monkeypatch):
    """A sub-second clip has no frame 1 s in; seeking there finds nothing.

    Answering "no frame" would mean software decode for a file hardware handles
    fine, so the seek is clamped to the last half-second of the file instead.
    """
    seek = _probe_seek(tmp_path, monkeypatch, frames=6)   # 0.24 s at 25 fps
    assert 0.0 <= seek < 6 / 25, seek
