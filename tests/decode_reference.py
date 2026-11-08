"""An independent decode of every frame, straight out of ffmpeg.

The tests that check *which* frame a capture hands out need a reference that
shares no code with the capture: this decodes the whole file twice with plain
ffmpeg/ffprobe calls (PTS from `ffprobe -show_frames`, pixels from a
`-f rawvideo` pipe) and zips the two.

No PyAV: the app decodes with the ffmpeg CLI, so the reference does too, and
nothing in the project depends on a second FFmpeg build.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from videocr import ffmpeg_capture

FFMPEG = ffmpeg_capture.FFMPEG
FFPROBE = ffmpeg_capture.FFPROBE


def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, check=True, **kwargs)


def probe_json(path) -> dict:
    """`ffprobe -show_format -show_streams` as JSON."""
    done = _run([FFPROBE, "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)], text=True)
    return json.loads(done.stdout)


def stream_start_time(path) -> float:
    """The video stream's own start_time (when its first frame arrives)."""
    stream = next(s for s in probe_json(path)["streams"] if s.get("codec_type") == "video")
    return float(stream.get("start_time") or 0.0)


def start_time(path) -> float:
    """The container's start_time: the earliest of all its streams."""
    return float(probe_json(path)["format"].get("start_time") or 0.0)


def frame_pts(path) -> list[float]:
    """Every video frame's PTS, in presentation order."""
    done = _run([FFPROBE, "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "frame=pts_time", "-print_format", "json", str(path)], text=True)
    frames = json.loads(done.stdout)["frames"]
    return [float(frame["pts_time"]) for frame in frames if "pts_time" in frame]


def raw_frames(path, width: int, height: int) -> list[np.ndarray]:
    """Every frame as BGR, decoded and converted by ffmpeg with no filter."""
    size = width * height * 3
    done = _run([FFMPEG, "-hide_banner", "-nostats", "-loglevel", "error", "-i", str(path),
                 "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
                 "-pix_fmt", "bgr24", "pipe:1"])
    data = done.stdout
    assert len(data) % size == 0, f"{path}: {len(data)} bytes is not whole frames of {width}x{height}"
    return [np.frombuffer(data[i:i + size], dtype=np.uint8).reshape(height, width, 3).copy()
            for i in range(0, len(data), size)]


def decode_reference(path) -> list[tuple[float, np.ndarray]]:
    """[(PTS, BGR frame)] for every frame, with strictly increasing PTS."""
    info = ffmpeg_capture._probe(str(path))
    pts = frame_pts(path)
    frames = raw_frames(path, info.width, info.height)
    assert len(pts) == len(frames), f"{path}: {len(pts)} PTS for {len(frames)} frames"
    assert all(a < b for a, b in zip(pts, pts[1:])), f"{path}: PTS not strictly increasing"
    return list(zip(pts, frames))


def every_frame(capture_cls, path) -> list[tuple[float, np.ndarray]]:
    """[(PTS, frame)] as a capture hands them out, reading to the end."""
    with capture_cls(str(path)) as cap:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((cap.get_last_pts(), frame.copy()))
    return frames


def encode(path: Path, *, size="320x240", rate="25", pix_fmt="yuv420p", codec="libx264",
           extra=(), frames: int | None = 100, video_start: float | None = None) -> Path:
    """A testsrc2 clip. `video_start` remuxes the video so it starts that long
    after an audio stream at 0 (a container start of 0 with a video offset).
    `frames=None` gives an infinite source, so it must be paired with `-t` in
    `extra`; the default 100 matches what every clip fixture wants."""
    gop = (["-x265-params", "keyint=10:min-keyint=10:log-level=error"]
           if codec == "libx265" else ["-g", "10"])
    common = ["-hide_banner", "-loglevel", "error", "-y"]
    source = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}"]
    limit = ["-frames:v", str(frames)] if frames is not None else []
    if video_start is None:
        _run([FFMPEG, *common, *source, *limit, "-pix_fmt", pix_fmt, "-c:v", codec, *gop,
              *extra, str(path)])
        return path
    # Encode video and audio both starting at 0 (with a millisecond encoder
    # time base, so the offset is not rounded to a frame), then remux with
    # the video input delayed. The container then starts at 0 and the video
    # stream later, which is the case a video-stream start_time gets wrong.
    plain = path.with_name(f"{path.stem}-plain{path.suffix}")
    length = (frames / 25 + 0.2) if frames is not None else 4.0
    _run([FFMPEG, *common, *source,
          "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
          "-map", "0:v", "-map", "1:a", *limit, "-t", str(length),
          "-pix_fmt", pix_fmt, "-c:v", codec, *gop, "-enc_time_base:v", "1:1000",
          "-c:a", "aac", *extra, str(plain)])
    _run([FFMPEG, *common, "-itsoffset", str(video_start), "-i", str(plain), "-i", str(plain),
          "-map", "0:v", "-map", "1:a", "-c", "copy", str(path)])
    plain.unlink()
    return path


def have_encoder(name: str) -> bool:
    done = subprocess.run([str(FFMPEG), "-hide_banner", "-encoders"], capture_output=True, text=True)
    return any(line.split()[1:2] == [name] for line in done.stdout.splitlines())
