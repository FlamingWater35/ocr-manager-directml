"""The decoder guard: no run may silently decode with a different backend.

The app ships ffmpeg and ffprobe in its own bin/ directory and decodes through
them, so a build where either binary is missing would fall back to an
estimate-based OpenCV reader that produces subtitles that do not line up with
the picture. `assert_reference_backend()` makes that a refusal, with one
explicit escape hatch, and `videocr.api.get_subtitles()` -- the only route a
pipeline run takes -- calls it before decoding a single frame.
"""
import pytest

from videocr import ffmpeg_capture


def test_ffmpeg_is_available():
    assert ffmpeg_capture.FFMPEG_AVAILABLE, (
        f"ffmpeg/ffprobe not found on PATH: {ffmpeg_capture.FFMPEG_ERROR}"
    )


def test_backend_is_ffmpeg():
    assert ffmpeg_capture.capture_backend_name() == "ffmpeg"


def test_assert_reference_backend_raises_when_ffmpeg_is_missing(monkeypatch):
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_AVAILABLE", False)
    monkeypatch.setattr(
        ffmpeg_capture, "FFMPEG_ERROR", "ffprobe not found on PATH"
    )
    monkeypatch.delenv(ffmpeg_capture.ALLOW_FALLBACK_ENV, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        ffmpeg_capture.assert_reference_backend()

    message = str(exc_info.value)
    assert "ffmpeg" in message
    assert ffmpeg_capture.ALLOW_FALLBACK_ENV in message


def test_assert_reference_backend_allows_escape_hatch(monkeypatch):
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_AVAILABLE", False)
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_ERROR", "boom")
    monkeypatch.setenv(ffmpeg_capture.ALLOW_FALLBACK_ENV, "1")

    assert ffmpeg_capture.assert_reference_backend() is None


@pytest.mark.parametrize("env_value", [None, "1"])
def test_assert_reference_backend_ok_when_ffmpeg_is_available(monkeypatch, env_value):
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_AVAILABLE", True)
    if env_value is None:
        monkeypatch.delenv(ffmpeg_capture.ALLOW_FALLBACK_ENV, raising=False)
    else:
        monkeypatch.setenv(ffmpeg_capture.ALLOW_FALLBACK_ENV, env_value)

    assert ffmpeg_capture.assert_reference_backend() is None


class _NeverReached(Exception):
    """Raised by a stubbed Video to prove the guard ran before any decode."""


def _stub_video(monkeypatch):
    from videocr import api

    def boom(*args, **kwargs):
        raise _NeverReached

    monkeypatch.setattr(api, "Video", boom)
    return api


def test_production_entry_point_refuses_when_ffmpeg_is_missing(monkeypatch):
    """`assert_reference_backend` must have an owner on the real OCR path.

    videocr.api.get_subtitles is the only route a pipeline run takes. With the
    check living solely in tools/fidelity_check, a missing ffmpeg silently
    demoted every run to the non-bit-exact, PTS-estimating fallback.
    """
    api = _stub_video(monkeypatch)
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_AVAILABLE", False)
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_ERROR", "boom")
    monkeypatch.delenv(ffmpeg_capture.ALLOW_FALLBACK_ENV, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        api.get_subtitles("no-such-video.mkv")

    assert ffmpeg_capture.ALLOW_FALLBACK_ENV in str(exc_info.value)


def test_production_entry_point_proceeds_with_escape_hatch(monkeypatch):
    """The opt-in still works: the guard passes and the run gets as far as
    opening the video (here, the stub that stands in for it)."""
    api = _stub_video(monkeypatch)
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_AVAILABLE", False)
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_ERROR", "boom")
    monkeypatch.setenv(ffmpeg_capture.ALLOW_FALLBACK_ENV, "1")

    with pytest.raises(_NeverReached):
        api.get_subtitles("no-such-video.mkv")


def test_refusal_does_not_leave_a_truncated_ass_file(tmp_path, monkeypatch):
    """save_subtitles_to_file opened 'w+' before doing any work, so anything
    that refused to run -- the decoder guard, an unbuildable filter chain --
    truncated or created a zero-byte .ass next to the video. An empty file
    reads as 'OCR produced nothing', not 'OCR did not run'."""
    from videocr import api

    out = tmp_path / "episode.ass"
    out.write_text("previous run's output", encoding="utf-8")

    api_module = _stub_video(monkeypatch)
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_AVAILABLE", False)
    monkeypatch.setattr(ffmpeg_capture, "FFMPEG_ERROR", "boom")
    monkeypatch.delenv(ffmpeg_capture.ALLOW_FALLBACK_ENV, raising=False)

    with pytest.raises(RuntimeError):
        api_module.save_subtitles_to_file("no-such-video.mkv", file_path=str(out))

    assert out.read_text(encoding="utf-8") == "previous run's output"

    missing = tmp_path / "fresh.ass"
    with pytest.raises(RuntimeError):
        api.save_subtitles_to_file("no-such-video.mkv", file_path=str(missing))
    assert not missing.exists()
