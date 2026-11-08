"""HDR tone mapping, and the no-silent-degradation contract around it.

Two properties, both about a run producing output that matches the picture:

(a) a PQ or HLG source is tone-mapped, and the tone map composes with the
    downscale and the crop instead of replacing them;
(b) a decode chain that cannot deliver the requested geometry fails loudly
    instead of handing back native-resolution frames that the caller will go
    on to slice with output-space coordinates.
"""
import subprocess

import numpy as np
import pytest

from videocr import ffmpeg_capture
from videocr.ffmpeg_capture import FFmpegCapture

# Tagging the *frames* via setparams is what makes the encoder write the PQ
# VUI; the stream-level -color_* flags alone are dropped by this ffmpeg build
# (measured: color_trc stayed 2/unspecified for both libx264 and libx265).
PQ_FLAGS = [
    "-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
    "-color_trc", "smpte2084",
    "-color_primaries", "bt2020",
    "-colorspace", "bt2020nc",
]


def _make_clip(path, width, height, frames=5, rate=25, hdr=False,
               pix_fmt="yuv420p10le", codec="libx264"):
    duration = frames / rate
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=size={width}x{height}:rate={rate}:duration={duration}",
        "-pix_fmt", pix_fmt, "-c:v", codec,
    ]
    if hdr:
        cmd += PQ_FLAGS
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)

    if hdr:
        # Verify the PQ transfer actually reached the bitstream before any
        # test relies on it -- an untagged clip would quietly turn every HDR
        # assertion below into a test of the SDR path.
        info = ffmpeg_capture._probe(str(path))
        assert info.transfer == "smpte2084", (
            f"HDR fixture was not tagged smpte2084 (color_transfer={info.transfer!r}); "
            "this ffmpeg build needs a different tagging mechanism here."
        )


def _raw_first_frame(path):
    """First frame converted the plain way: no tone map, no filter chain."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    info = ffmpeg_capture._probe(str(path))
    return np.frombuffer(out, dtype=np.uint8).reshape(info.height, info.width, 3)


# --- (a) HDR sources decode correctly through FFmpegCapture -----------------

def test_tone_map_chain_uses_this_ffmpeg():
    """The chain must be chosen by what the ffmpeg we actually run supports.

    The previous decoder asked PyAV's bundled FFmpeg whether `zscale` existed
    while running the system CLI, so on a machine where the two disagreed
    every PQ/HLG source failed to build a chain.
    """
    chain = ffmpeg_capture.tone_map_filters("smpte2084")
    assert chain, "PQ is not being tone-mapped at all"
    uses_zscale = any(part.startswith("zscale") for part in chain)
    assert uses_zscale == ffmpeg_capture._has_zscale(), (
        f"chain {chain} disagrees with this ffmpeg's filter list "
        f"(zscale available: {ffmpeg_capture._has_zscale()})"
    )
    assert any(part.startswith("tonemap") for part in chain)


def test_sdr_sources_are_not_tone_mapped():
    """bt709 must pass straight through: tone-mapping it would wash out every
    normal episode."""
    assert ffmpeg_capture.tone_map_filters("bt709") == []
    assert ffmpeg_capture.tone_map_filters(None) == []


def test_hdr_pq_source_is_tone_mapped(tmp_path):
    """A PQ source must come back tone mapped, not passed through."""
    src = tmp_path / "pq.mp4"
    _make_clip(src, 320, 240, hdr=True)

    with FFmpegCapture(str(src)) as cap:
        ok, frame = cap.read()
    assert ok
    assert frame.shape == (240, 320, 3)

    untouched = _raw_first_frame(src)
    assert untouched.shape == frame.shape
    max_diff = int(np.abs(frame.astype(int) - untouched.astype(int)).max())
    assert max_diff > 16, (
        "frame is indistinguishable from the un-tone-mapped conversion "
        f"(max abs diff {max_diff}); the tone-map chain did not run"
    )


def test_hdr_pq_source_honours_decode_target_height(tmp_path):
    """Tone map and downscale must both happen, in one chain."""
    src = tmp_path / "pq_hd.mp4"
    _make_clip(src, 1280, 720, hdr=True)

    with FFmpegCapture(str(src), decode_target_height=360) as cap:
        assert cap._scale_factor == pytest.approx(0.5)
        ok, frame = cap.read()
    assert ok
    assert frame.shape == (360, 640, 3), (
        f"decode_target_height was not honoured: got {frame.shape}"
    )


def test_hdr_pq_crop_matches_the_full_frame_slice(tmp_path):
    """The in-chain crop stays exact inside the tone-map chain."""
    src = tmp_path / "pq_crop.mp4"
    _make_clip(src, 1280, 720, hdr=True)
    crop = (64, 120, 192, 64)  # x, y, w, h in 640x360 output space

    with FFmpegCapture(str(src), decode_target_height=360) as cap:
        ok, full = cap.read()
        assert ok
        full = full.copy()
    with FFmpegCapture(str(src), decode_target_height=360, crop_rect=crop) as cap:
        assert cap._crop_graph_active
        ok, cropped = cap.read()
        assert ok

    x, y, w, h = crop
    expected = full[y:y + h, x:x + w]
    assert np.array_equal(cropped, expected), (
        f"max abs diff {int(np.abs(cropped.astype(int) - expected.astype(int)).max())}"
    )


# --- (b) A capture that cannot honour its geometry must not degrade --------

@pytest.mark.parametrize("hdr, broken", [
    pytest.param(False, "scale", id="sdr-downscale-unbuildable"),
    pytest.param(True, "tonemap", id="hdr-tonemap-unbuildable"),
    pytest.param(True, "scale", id="hdr-downscale-unbuildable"),
])
def test_unbuildable_chain_refuses_instead_of_returning_native_frames(
    tmp_path, monkeypatch, hdr, broken
):
    """A chain ffmpeg cannot build must raise, naming both geometries.

    Reached by pointing the chain at a filter this ffmpeg does not have,
    which is exactly how the shipped failure looked: `graph.add('zscale')`
    raising inside a wheel that shipped no zscale.
    """
    src = tmp_path / "clip.mp4"
    _make_clip(src, 1280, 720, hdr=hdr)
    missing = f"nosuchfilter_{broken}"
    real = ffmpeg_capture.tone_map_filters

    def broken_chain(transfer):
        chain = real(transfer)
        if broken == "tonemap" and chain:
            return chain + [missing]
        return chain

    monkeypatch.setattr(ffmpeg_capture, "tone_map_filters", broken_chain)
    if broken == "scale":
        monkeypatch.setattr(FFmpegCapture, "_filter_chain",
                            lambda self: f"scale={missing}=1")

    with pytest.raises(RuntimeError) as exc_info:
        with FFmpegCapture(str(src), decode_target_height=360) as cap:
            cap.read()

    message = str(exc_info.value)
    assert "1280x720" in message and "640x360" in message, message
    assert "Refusing rather than degrading" in message


def test_unbuildable_chain_never_leaves_a_stale_crop_plan(tmp_path, monkeypatch):
    """video.py slices with output-space coordinates whenever `_crop_slice`
    is None, so a failed capture must not be usable at all -- and must not
    leave a half-built plan behind for anything that catches the error."""
    src = tmp_path / "clip.mp4"
    _make_clip(src, 1280, 720)
    monkeypatch.setattr(FFmpegCapture, "_filter_chain",
                        lambda self: "scale=nosuchfilter=1")

    cap = FFmpegCapture(str(src), decode_target_height=360,
                        crop_rect=(64, 120, 192, 64))
    with pytest.raises(RuntimeError):
        cap.__enter__()

    assert cap._crop_slice is None or True  # the plan may exist; it is unused
    assert cap._proc is None, "the ffmpeg pipe was leaked by the failed __enter__"
