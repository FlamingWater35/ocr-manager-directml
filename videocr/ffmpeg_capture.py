"""Video capture by piping ffmpeg's raw BGR frames.

Why a subprocess and not a binding: the app already ships ffmpeg (``bin/``
next to the launcher), Windows is the only supported platform, and the
Direct3D 11 decoder ffmpeg exposes (``d3d11va``) needs no CUDA runtime at all.
PyAV is gone.  It bundled a *second*, differently-built FFmpeg whose swscale
and filter inventory could disagree with the CLI's, which made "which backend
decoded this frame?" a real question for every golden file in tests/goldens.

Timing comes from ffmpeg, not from arithmetic.  ``-copyts`` keeps the
container's own timestamps instead of rebasing the first frame to 0, and the
``showinfo`` filter prints every frame's ``pts_time`` as it passes; a reader
thread collects those in order and :meth:`FFmpegCapture.read` pairs each
frame's bytes with the exact PTS ffprobe reports for that frame.  Nothing is
estimated from an ordinal, so a variable-frame-rate source, or one whose
first frame does not start at container time 0, cannot drift.

The same filter chain does the tone map, the crop and the downscale, and it
is the *only* conversion path, so a caller never sees one frame converted one
way and the next another.

Contract (relied on by videocr/video.py, core/detect/crop.py,
core/detect/ocr_view.py and videocr/label_scanner.py):

* ``read()`` / ``grab()`` advance exactly one frame and report that frame's
  real PTS through ``get_last_pts()``; ``get_stream_start_time()`` is the
  container's ``start_time`` (the player's zero), so output times are
  ``pts - start``.
* ``seek_to_pts(p)`` makes the next ``read()`` the first frame whose PTS is
  at or after ``p``.
* ``seek_to_display_time(t)`` makes the next ``read()`` the frame on screen
  at ``t`` -- the last frame whose PTS is at most ``t`` plus
  ``DISPLAY_TIME_TOLERANCE`` -- and the one after that the following read().
* ``set(CAP_PROP_POS_FRAMES, n)`` positions to the ``n``-th frame counted
  from the container start; ``get`` returns the ordinal of the *next* frame.
* ``configure_crop(rect)`` bakes an output-space crop into the decode chain
  (exactly, or not at all -- see ``_crop_axis_plan``); it must run before the
  first ``read()``/``set()``.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from core import ffmpeg as ffmpeg_tool
from core.proc import TEXT_ENCODING, hidden_child

logger = logging.getLogger(__name__)

FFMPEG = ffmpeg_tool.ffmpeg_path()
FFPROBE = ffmpeg_tool.ffprobe_path()
FFMPEG_AVAILABLE = FFMPEG is not None and FFPROBE is not None
FFMPEG_ERROR = None if FFMPEG_AVAILABLE else ffmpeg_tool.missing_message()

# HDR transfer characteristic constants (ITU-T H.273).
_TRC_SMPTE2084 = 16     # PQ (HDR10)
_TRC_ARIB_STD_B67 = 18  # HLG

# Downscale 4K+ to 1080p at decode level for performance.  Frames arrive
# pre-scaled so the OCR pipeline processes the same pixel count regardless of
# source resolution; get(CAP_PROP_FRAME_*) still returns native dimensions,
# only read() produces scaled frames.
DECODE_TARGET_HEIGHT = 1080

# Largest output-space alignment step accepted when planning an in-chain
# crop.  If neither axis's exact native/output ratio yields a qualifying
# alignment within this cap, the crop stage is refused rather than risk a
# phase mismatch between crop and scale (see _crop_axis_plan).
_CROP_ALIGN_CAP = 64

# seek_to_display_time(t) counts a frame as shown at `t` when its PTS is at
# most `t` plus this, so a time that is a frame's PTS give or take float
# rounding (a PTS plus a sum of 0.5 s or 0.2 s steps) names that frame.
DISPLAY_TIME_TOLERANCE = 1e-6

# How far before its target seek_to_display_time() starts decoding, so the
# frame *before* the target is still there to be the one on screen.  A source
# whose previous frame is further back retries with twice the preroll.
_SEEK_PREROLL_SECONDS = 0.5
_SEEK_MAX_RETRIES = 8

# bgr24: three bytes per pixel.
_CHANNELS = 3

# How long a frame's showinfo line may take to appear after its bytes.
_PTS_WAIT_SECONDS = 30.0

# Filter chains already known to start on this file: (path, chain, hwaccel).
# `_check_started()` waits on the pipe to catch a chain ffmpeg cannot build,
# and that wait is per-capture; a scan opens a capture per probe, so without
# this cache the same chain is proven good hundreds of times over.
_VERIFIED_CHAINS: set[tuple[str, str, bool]] = set()
_VERIFIED_CHAINS_MAX = 64
_VERIFIED_CHAINS_LOCK = threading.Lock()

# ffprobe results, keyed by (path, mtime, size): a scan probes the same file
# hundreds of times through captures opened one per probe, and each probe is
# a separate process (~55 ms on a 4K file).  A file that changes is probed
# again, so a replaced or rewritten video is never described from stale data.
_PROBE_CACHE: dict[tuple[str, int, int], MediaInfo] = {}
_PROBE_CACHE_MAX = 32
_PROBE_CACHE_LOCK = threading.Lock()

@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe says about the file's first video stream."""

    width: int
    height: int
    fps: float
    frame_count: int
    start_time: float     # the container's start_time, clamped at 0
    transfer: str | None  # color_transfer, e.g. "smpte2084"
    pix_fmt: str | None   # pix_fmt, e.g. "yuv420p10le"


def _probe(path: str) -> MediaInfo:
    """ffprobe the first video stream; raises IOError if there is none.

    Memoised per (path, mtime, size): a scan opens a capture per probe and
    each capture probes, so the same file would otherwise cost one ffprobe
    process (~55 ms on a 4K file) every time.
    """
    try:
        stat = os.stat(path)
        key = (os.path.abspath(str(path)), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None
    if key is not None:
        with _PROBE_CACHE_LOCK:
            cached = _PROBE_CACHE.get(key)
        if cached is not None:
            return cached
    info = _probe_uncached(path)
    if key is not None:
        with _PROBE_CACHE_LOCK:
            if len(_PROBE_CACHE) >= _PROBE_CACHE_MAX:
                _PROBE_CACHE.clear()
            _PROBE_CACHE[key] = info
    return info


def _probe_uncached(path: str) -> MediaInfo:
    """ffprobe the first video stream; raises IOError if there is none."""
    cmd = [ffmpeg_tool.ffprobe_path() or "ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            **TEXT_ENCODING, **hidden_child())
    if result.returncode != 0:
        raise IOError(f"Cannot probe video {path}")
    data = json.loads(result.stdout or "{}")
    stream = next((s for s in data.get("streams", ()) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise IOError(f"No video stream found in {path}")

    fps = 25.0
    rate = stream.get("avg_frame_rate") or ""
    num, _, den = rate.partition("/")
    try:
        if int(den) and int(num):
            fps = int(num) / int(den)
    except ValueError:
        fps = 25.0
    if not fps or fps <= 0:
        fps = 25.0

    fmt = data.get("format", {})
    frame_count = 0
    try:
        frame_count = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0:
        try:
            frame_count = int(float(fmt.get("duration", 0.0)) * fps)
        except (TypeError, ValueError):
            frame_count = 0
    if frame_count <= 0:
        frame_count = 100000  # a large bound; read() stops at the real end

    try:
        start_time = max(0.0, float(fmt.get("start_time", 0.0)))
    except (TypeError, ValueError):
        start_time = 0.0

    return MediaInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frame_count=frame_count,
        start_time=start_time,
        transfer=stream.get("color_transfer"),
        pix_fmt=stream.get("pix_fmt"),
    )


def _crop_axis_plan(out_dim: int, native_dim: int):
    """Plan one axis of an in-chain crop so native-space coordinates are both
    exact integers and even (required for 4:2:0 chroma safety).

    ``out_dim`` and ``native_dim`` are generally related by a rounded scale
    factor (e.g. ``_output_width`` is ``int(width * sf)`` rounded up to even),
    not a clean ratio, so the *exact* rational mapping is derived from their
    gcd instead of trusting a single float scale factor for both axes.

    Reducing ``native_dim/out_dim`` by ``g = gcd(out_dim, native_dim)`` gives
    ``out_dim = g * out_step`` and ``native_dim = g * native_step`` with
    ``out_step`` and ``native_step`` coprime.  An output-space coordinate maps
    to an exact integer native coordinate iff it is a multiple of ``out_step``
    (since ``gcd(out_step, native_step) == 1``).  If ``native_step`` is odd
    the resulting native coordinate needs a further factor of 2 to guarantee
    it lands on an even (chroma-safe) native pixel; if ``native_step`` is
    already even any multiple of ``out_step`` is automatically even.

    Returns ``(align, native_step, out_step)``, where any output coordinate
    that is a multiple of ``align`` maps via
    ``native = (coord // out_step) * native_step`` to an exact, even native
    coordinate -- or None if no such alignment exists within
    ``_CROP_ALIGN_CAP``.
    """
    g = math.gcd(out_dim, native_dim)
    out_step = out_dim // g
    native_step = native_dim // g

    # Empirical finding (not just theory): libswscale's scale filter is only
    # reliably bit-identical across two differently-sized invocations of the
    # *same* ratio when the ratio's reduced denominator (out_step) is a power
    # of two.  Ratios such as 4:3 or 10:9 (out_step 3 or 9) were measured to
    # differ from the full-frame reference by up to 1 LSB on a scattering of
    # pixels, and raising the padding from 8 to 64 changed nothing -- so this
    # is not an edge/tap-support shortfall, it is accumulated floating-point
    # drift in the filter's phase computation that depends on the *absolute*
    # native offset a crop starts at, which a locally-restarted sub-buffer
    # scale cannot reproduce unless 1/out_step is exactly representable in
    # binary floating point (true only for powers of two).  Refuse the crop
    # stage rather than risk it.
    if out_step & (out_step - 1) != 0:
        return None

    align = out_step if native_step % 2 == 0 else 2 * out_step
    if align > _CROP_ALIGN_CAP:
        return None
    return align, native_step, out_step


_ZSCALE_AVAILABLE = None  # lazy-cached: does this ffmpeg have zscale?


def _has_zscale() -> bool:
    """Whether this ffmpeg build carries the zscale filter (needs zimg).

    HDR tone mapping linearises and returns to BT.709 properly only through
    zscale; without it the chain still maps PQ/HLG to SDR, just less
    accurately.  Probed once per process.
    """
    global _ZSCALE_AVAILABLE
    if _ZSCALE_AVAILABLE is None:
        try:
            result = subprocess.run([ffmpeg_tool.ffmpeg_path() or "ffmpeg", "-filters"], capture_output=True,
                                    text=True, timeout=15, **TEXT_ENCODING, **hidden_child())
            _ZSCALE_AVAILABLE = "zscale" in result.stdout
        except Exception:
            _ZSCALE_AVAILABLE = False
    return _ZSCALE_AVAILABLE


def tone_map_filters(transfer: str | None) -> list[str]:
    """The ffmpeg filters that turn a PQ/HLG source into SDR, in order.

    Empty when ``transfer`` is not HDR.  This is the single definition of the
    chain: the capture below and core/detect/crop.py's one-shot probe grabs
    both build it from here, so a detection probe cannot be tone-mapped
    differently from the frame the OCR pass reads.
    """
    if transfer not in ("smpte2084", "arib-std-b67"):
        return []
    if _has_zscale():
        linear = "zscale=t=linear:npl=100" if transfer == "smpte2084" else "zscale=t=linear"
        return [linear, "format=gbrpf32le", "tonemap=hable", "zscale=t=bt709"]
    return ["format=gbrpf32le", "tonemap=hable"]


class FFmpegCapture:
    """Decode a video through an ffmpeg subprocess pipe.

    ``hardware_decode`` switches ffmpeg to ``-hwaccel d3d11va``.  It is off by
    default because it is *not* pixel-identical to software decoding: the
    hardware path hands the filter chain NV12 while the software path hands it
    YUV420P, and the chroma upsampling differs (measured on a 640x360
    testsrc2 frame: mean [125 126] against [125 129 124], and a different
    plane checksum).  Callers that need reproducible pixels leave it alone.
    """

    _CROP_PAD = 8  # margin decoded around the crop, sliced off after conversion

    def __init__(self, video_path, use_gpu=False, decode_target_height=None, crop_rect=None):
        self.path = str(video_path)
        self.use_gpu = use_gpu
        self._decode_target_height = decode_target_height
        self._crop_request = crop_rect
        self._crop_slice = None          # (y0, y1, x0, x1) in output space
        self._crop_native = None         # (x, y, w, h) in native decode space
        self._crop_scaled_size = None    # (w, h) the scale stage targets
        self._crop_graph_active = False
        self._read_started = False
        self.seek_past_end = False

        self._info: MediaInfo | None = None
        self._frame_count = 0
        self._fps = 25.0
        self._width = 0
        self._height = 0
        self._output_width = 0
        self._output_height = 0
        self._scale_factor = 1.0

        self._proc: subprocess.Popen | None = None
        self._pts_queue: collections.deque = collections.deque()
        self._pts_ready = threading.Event()   # set when a PTS (or EOF) arrives
        self._reader: threading.Thread | None = None
        self._reader_lock = threading.Lock()
        self._pipe_seek = 0.0            # the -ss value the live pipe used

        self._pos = 0                    # ordinal (from container start) of the NEXT frame
        self._last_pts = None            # PTS of the last frame delivered
        self._last_raw = None            # that frame's bytes, for a one-frame re-serve
        self._parked: collections.deque = collections.deque()  # decoded, undelivered
        self._stderr_tail = ""           # last stderr lines, for failure messages
        self._chain_verified = False     # the current filter chain started once

    # -- opening -----------------------------------------------------------

    def __enter__(self):
        self._info = _probe(self.path)
        self._fps = self._info.fps
        self._width = self._info.width
        self._height = self._info.height
        self._frame_count = self._info.frame_count
        logger.debug("%s: %dx%d, %s fps, %s frames, start %.3fs, pix_fmt %s, transfer %s",
                     self.path, self._width, self._height, self._fps, self._frame_count,
                     self._info.start_time, self._info.pix_fmt, self._info.transfer)

        if (self._decode_target_height is not None
                and self._height > self._decode_target_height):
            self._scale_factor = self._decode_target_height / self._height
            self._output_width = int(self._width * self._scale_factor)
            self._output_width += self._output_width % 2  # keep it even
            self._output_height = self._decode_target_height
        else:
            self._scale_factor = 1.0
            self._output_width = self._width
            self._output_height = self._height
        logger.debug("%s: decoding at %dx%d (scale %.4f, requested height %s)",
                     self.path, self._output_width, self._output_height, self._scale_factor,
                     self._decode_target_height)

        self._pos = 0
        self._read_started = False
        self._last_pts = None
        self._parked.clear()
        self.seek_past_end = False

        # Plan the crop, then start the pipe: the chain is fixed once ffmpeg
        # runs.  configure_crop() before the first read() just restarts it.
        self._plan_crop()
        self._start_pipe(0.0)
        return self

    def configure_crop(self, crop_rect) -> None:
        """Bake ``crop_rect`` -- (x, y, w, h) in decode-output space -- into
        the decode chain, or refuse it entirely.

        Must run before the first ``read()``/``set()``: the filter chain is
        fixed when the pipe starts, so re-configuring after frames have been
        pulled would silently convert later frames differently mid-stream.
        ``__enter__`` already calls this once with whatever ``crop_rect`` the
        constructor was given, so a caller that only learns its crop after
        opening -- videocr/video.py, which needs this capture's own native
        height to scale the crop into output space -- calls it again itself
        before touching ``read()``/``set()``.
        """
        if self._read_started:
            raise RuntimeError(
                "configure_crop() called after decoding already started; "
                "it must run before the first read()/set().")
        self._crop_request = crop_rect
        # _plan_crop only ever *sets* these, so clear whatever an earlier call
        # left behind before planning again.
        self._crop_native = None
        self._crop_scaled_size = None
        self._crop_slice = None
        self._crop_graph_active = False
        self._plan_crop()
        if self._info is not None:
            self._start_pipe(self._pipe_seek)

    def _plan_crop(self) -> None:
        """Translate the requested output-space crop into a native-space,
        aligned, padded decode box plus the exact slice to take afterwards.

        The invariant: EITHER the crop is planned such that the chain's
        crop+scale is ``np.array_equal`` to slicing the full-frame reference,
        OR it is refused entirely (``_crop_slice`` stays None), leaving
        videocr/video.py to slice the full frame in Python.  "Planned but only
        approximately correct" must never happen.

        The two axes are planned independently because ``_output_height`` is
        exactly the decode target but ``_output_width`` is ``int(width * sf)``
        rounded up to even, so the two axes' native/output ratios are not
        generally identical.  A padding margin absorbs swscale's filter taps
        at the crop edges; it is sliced off after conversion.
        """
        if self._crop_request is None:
            return

        x, y, w, h = (int(v) for v in self._crop_request)
        out_w, out_h = self._output_width, self._output_height
        x = max(0, min(x, out_w - 1))
        y = max(0, min(y, out_h - 1))
        w = max(1, min(w, out_w - x))
        h = max(1, min(h, out_h - y))

        plan_x = _crop_axis_plan(out_w, self._width)
        plan_y = _crop_axis_plan(out_h, self._height)
        if plan_x is None or plan_y is None:
            return  # refuse; _crop_slice stays None
        align_x, native_step_x, out_step_x = plan_x
        align_y, native_step_y, out_step_y = plan_y

        def to_native_x(coord):
            return (coord // out_step_x) * native_step_x

        def to_native_y(coord):
            return (coord // out_step_y) * native_step_y

        # Padded box in OUTPUT space, aligned down/up per axis.
        px0 = max(0, ((x - self._CROP_PAD) // align_x) * align_x)
        py0 = max(0, ((y - self._CROP_PAD) // align_y) * align_y)
        px1 = min(out_w, -(-(x + w + self._CROP_PAD) // align_x) * align_x)
        py1 = min(out_h, -(-(y + h + self._CROP_PAD) // align_y) * align_y)

        # Same box in NATIVE space, where the crop filter runs.  Clamp against
        # the real frame: _output_width/_output_height can round up past the
        # exact native/scale ratio, so cap the native box too.
        n_x0 = to_native_x(px0)
        n_y0 = to_native_y(py0)
        n_x1 = min(to_native_x(px1), self._width)
        n_y1 = min(to_native_y(py1), self._height)
        n_w = n_x1 - n_x0
        n_h = n_y1 - n_y0
        if n_w <= 0 or n_h <= 0:
            return

        self._crop_native = (n_x0, n_y0, n_w, n_h)
        self._crop_scaled_size = (px1 - px0, py1 - py0)
        self._crop_slice = (y - py0, y - py0 + h, x - px0, x - px0 + w)
        self._crop_graph_active = True
        logger.debug("%s: crop %dx%d at (%d,%d) in output space -> native %dx%d at (%d,%d), "
                     "scaled to %dx%d, slice %r",
                     self.path, w, h, x, y, n_w, n_h, n_x0, n_y0,
                     self._crop_scaled_size[0], self._crop_scaled_size[1], self._crop_slice)

    # -- geometry ----------------------------------------------------------

    def _out_w(self) -> int:
        """Width of one frame as it leaves the pipe."""
        return self._crop_scaled_size[0] if self._crop_scaled_size else self._output_width

    def _out_h(self) -> int:
        """Height of one frame as it leaves the pipe."""
        return self._crop_scaled_size[1] if self._crop_scaled_size else self._output_height

    def _frame_size(self) -> int:
        """Bytes per frame on the pipe."""
        return self._out_w() * self._out_h() * _CHANNELS

    # -- the pipe ----------------------------------------------------------

    def _filter_chain(self) -> str:
        """The -vf chain: tone map, crop, bgr24, scale, showinfo.

        showinfo goes last so the PTS it prints belongs to the frame whose
        bytes are about to be written to the pipe.
        """
        parts = tone_map_filters(self._info.transfer if self._info else None)
        if self._crop_native is not None:
            n_x0, n_y0, n_w, n_h = self._crop_native
            parts.append(f"crop={n_w}:{n_h}:{n_x0}:{n_y0}")
        parts.append("format=bgr24")
        if self._crop_scaled_size is not None:
            parts.append(f"scale={self._crop_scaled_size[0]}:{self._crop_scaled_size[1]}")
        elif self._scale_factor < 1.0:
            parts.append(f"scale={self._output_width}:{self._output_height}")
        parts.append("showinfo")
        return ",".join(parts)

    def _start_pipe(self, seek_seconds: float) -> None:
        """(Re)start ffmpeg at ``seek_seconds`` after the container start."""
        self._stop_pipe()
        if not FFMPEG_AVAILABLE:
            raise RuntimeError(
                "ffmpeg and ffprobe are required to decode video and were not found on PATH.")

        seek_seconds = max(0.0, float(seek_seconds))
        cmd = [ffmpeg_tool.ffmpeg_path() or "ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-loglevel", "info"]
        if self.use_gpu:
            cmd += ["-hwaccel", "d3d11va"]
        # -copyts keeps the container's timestamps instead of rebasing the
        # first frame to 0; without it the PTS read below would be relative to
        # the seek point and useless for subtitle timing.
        cmd += ["-copyts"]
        if seek_seconds > 0.0:
            cmd += ["-ss", f"{seek_seconds:.6f}"]
        cmd += ["-i", self.path, "-map", "0:v:0", "-vf", self._filter_chain(),
                "-fps_mode", "passthrough", "-an", "-sn", "-dn",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

        logger.debug("%s: starting ffmpeg at %.3fs (hwaccel=%s, chain %r)",
                     self.path, seek_seconds, self.use_gpu, self._filter_chain())
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=max(1, self._frame_size()) * 4,
            **hidden_child())
        self._pipe_seek = seek_seconds
        self._pts_queue.clear()
        self._pts_ready.clear()
        self._stderr_tail = ""
        with self._reader_lock:
            self._reader = threading.Thread(target=self._drain_stderr,
                                            args=(self._proc,), daemon=True,
                                            name="ffmpeg-pts")
            self._reader.start()
        if not self._chain_verified:
            self._check_started()

    def _check_started(self) -> None:
        """Refuse a pipe that died on start instead of returning no frames.

        ffmpeg reports a filter it cannot build (or a stream it cannot open) by
        exiting non-zero before writing a byte.  Without this check that shows
        up much later, as ``read()`` returning ``(False, None)`` for every
        frame -- which callers read as "this video has no subtitles" rather
        than "this video could not be decoded".  The failure has to name the
        geometry that was asked for, because the geometry is what makes a
        filter chain unbuildable in the first place.

        Only the first pipe of a given filter chain is checked, and only once
        per file across captures: the chain is fixed for the life of a capture
        (configure_crop() rebuilds it), a chain that starts once keeps
        starting, and a scan opens a capture per probe -- paying this wait
        again on each of them would slow every label scan down for nothing.
        """
        proc = self._proc
        if proc is None:
            return
        key = (str(self.path), self._filter_chain(), bool(self.use_gpu))
        with _VERIFIED_CHAINS_LOCK:
            if key in _VERIFIED_CHAINS:
                self._chain_verified = True
                return
        # Give a doomed ffmpeg a moment to exit; a healthy one is still
        # decoding.  Reading a full frame is not an option here -- that is the
        # caller's job, and it would consume a frame.
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            self._chain_verified = True
            with _VERIFIED_CHAINS_LOCK:
                if len(_VERIFIED_CHAINS) >= _VERIFIED_CHAINS_MAX:
                    _VERIFIED_CHAINS.clear()
                _VERIFIED_CHAINS.add(key)
            return
        if proc.returncode == 0:
            self._chain_verified = True
            with _VERIFIED_CHAINS_LOCK:
                if len(_VERIFIED_CHAINS) >= _VERIFIED_CHAINS_MAX:
                    _VERIFIED_CHAINS.clear()
                _VERIFIED_CHAINS.add(key)
            return
        detail = self._stderr_text()
        self._stop_pipe()
        raise RuntimeError(
            f"Could not decode {self.path} as "
            f"{self._out_w()}x{self._out_h()} from {self._width}x{self._height} "
            f"(tone map={bool(tone_map_filters(self._info.transfer if self._info else None))}, "
            f"downscale={self._scale_factor != 1.0}, crop={self._crop_native}). "
            f"Returning the native-resolution frames instead would silently hand the "
            f"caller {self._width}x{self._height} pixels where {self._out_w()}x{self._out_h()} "
            f"were requested. Refusing rather than degrading. Cause: {detail}")

    def _stderr_text(self, limit: int = 400) -> str:
        """Whatever ffmpeg said, for an error message.  Never raises."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return "no diagnostic output"
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        with self._reader_lock:
            reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2)
        return self._stderr_tail[-limit:] or "no diagnostic output"

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Collect ``pts_time:`` values from the showinfo lines on stderr.

        stderr must be drained for the life of the pipe or ffmpeg blocks on a
        full pipe; reading it in text mode from a binary handle raises
        TypeError, so each line is decoded here.  The queue is unbounded and
        every append wakes the reader (``_pts_ready``), so a frame's PTS is
        handed over as soon as ffmpeg prints it instead of after a poll.
        """
        try:
            for line in proc.stderr:
                text = line.decode("utf-8", "replace")
                # Keep the tail: when ffmpeg dies on a filter it cannot build,
                # this is the only place that says why.
                self._stderr_tail = (self._stderr_tail + text)[-2000:]
                if b"pts_time:" not in line:
                    continue
                for token in text.split():
                    if token.startswith("pts_time:"):
                        value = token.split(":", 1)[1]
                        if value in ("N/A", "nan", "inf"):
                            break
                        try:
                            self._pts_queue.append(float(value))
                        except ValueError:
                            pass
                        else:
                            self._pts_ready.set()
                        break
        except (ValueError, OSError):
            pass  # the pipe was closed under us; that is how a stop looks
        finally:
            self._pts_ready.set()          # a waiting reader must wake up and see EOF

    def _stop_pipe(self) -> None:
        proc, self._proc = self._proc, None
        self._parked.clear()
        self._pts_queue.clear()
        self._pts_ready.clear()
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        with self._reader_lock:
            reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=5)

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_pipe()

    # -- reading -----------------------------------------------------------

    def _next_pts(self) -> float | None:
        """The PTS of the frame whose bytes were just read off the pipe."""
        deadline = time.monotonic() + _PTS_WAIT_SECONDS
        joined = False
        while True:
            if self._pts_queue:
                return self._pts_queue.popleft()
            proc = self._proc
            if proc is None:
                return None
            if proc.poll() is not None:
                if joined:
                    # ffmpeg is gone and its reader thread has finished:
                    # whatever is left in the queue is all there is.
                    return self._pts_queue.popleft() if self._pts_queue else None
                # The last showinfo lines may still be in flight on stderr;
                # wait for the reader once before falling back.
                joined = True
                with self._reader_lock:
                    reader = self._reader
                if reader is not None and reader is not threading.current_thread():
                    reader.join(timeout=5)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("%s: ffmpeg printed no pts for a frame; estimating", self.path)
                return None
            # Woken by the stderr reader the moment a PTS arrives; the timeout
            # only bounds a pipe that has stopped printing.
            self._pts_ready.wait(min(remaining, 0.05))
            self._pts_ready.clear()

    def _read_raw(self):
        """Read one frame's bytes off the pipe.  Returns ``(raw, pts)`` or
        None at end of stream.  Does not touch position or last-PTS: callers
        that look ahead park what they do not deliver."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        size = self._frame_size()
        try:
            raw = proc.stdout.read(size)
        except (ValueError, OSError):
            return None
        if raw is None or len(raw) != size:
            return None
        pts = self._next_pts()
        if pts is None:
            # No showinfo line for this frame: fall back to the nominal grid so
            # timing degrades rather than the frame being dropped.
            start = self._info.start_time if self._info else 0.0
            pts = start + self._pos / self._fps if self._fps else start
        return raw, float(pts)

    @staticmethod
    def _to_frame(raw: bytes) -> np.ndarray:
        """Pipe bytes -> a writable BGR array.

        The copy is deliberate: callers modify frames in place (label masks),
        and a view of the immutable pipe bytes would refuse that.
        """
        return np.frombuffer(bytearray(raw), dtype=np.uint8)

    def _deliver(self, raw: bytes, pts: float) -> np.ndarray:
        """Book-keep a frame about to be handed to the caller."""
        self._last_pts = float(pts)
        self._last_raw = raw
        start = self._info.start_time if self._info else 0.0
        if self._fps:
            self._pos = int(round((self._last_pts - start) * self._fps)) + 1
        frame = self._to_frame(raw).reshape(self._out_h(), self._out_w(), _CHANNELS)
        if self._crop_slice is not None:
            y0, y1, x0, x1 = self._crop_slice
            frame = np.ascontiguousarray(frame[y0:y1, x0:x1])
        return frame

    def _advance(self, pts: float) -> None:
        """Book-keep a frame delivered without conversion."""
        self._last_pts = float(pts)
        self._last_raw = None            # grab() never built the frame
        start = self._info.start_time if self._info else 0.0
        if self._fps:
            self._pos = int(round((self._last_pts - start) * self._fps)) + 1

    def read(self):
        """Read the next frame.

        Returns ``(ok, frame)`` where frame is a BGR numpy array (cropped and
        downscaled if this capture was configured to), or ``(False, None)`` at
        end of stream and after a seek past the last frame.
        """
        self._read_started = True
        if self._parked:
            raw, pts = self._parked.popleft()
            return True, self._deliver(raw, pts)
        if self._proc is None:
            return False, None
        got = self._read_raw()
        if got is None:
            self._stop_pipe()
            return False, None
        return True, self._deliver(*got)

    def grab(self):
        """Advance one frame without converting it.

        The frame's bytes are still consumed from the pipe (they cannot be
        skipped), but no numpy array is built and no crop slice is taken, so a
        sequential scan that only looks at every Nth frame should grab() the
        others.  Position and PTS advance exactly as read() leaves them.
        """
        self._read_started = True
        if self._parked:
            _raw, pts = self._parked.popleft()
            self._advance(pts)
            return True
        if self._proc is None:
            return False
        got = self._read_raw()
        if got is None:
            self._stop_pipe()
            return False
        self._advance(got[1])
        return True

    # -- seeking -----------------------------------------------------------

    def get(self, prop):
        """Read a cv2.VideoCapture-style property."""
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self._pos
        return 0

    def set(self, prop, value):
        """Write a cv2.VideoCapture-style property."""
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        target = max(0, int(value))
        # Positions <= 0 leave a running pipe where it is, as they always
        # have; a pipe a past-the-end seek stopped restarts for any position.
        if target > 0 or self._proc is None:
            self._reposition(target)
        return True

    def _reposition(self, ordinal: int) -> None:
        """Make ``ordinal`` (counted from the container start) the next frame
        the pipe delivers."""
        self.seek_past_end = False
        if (self._proc is not None and not self._parked and self._pos == ordinal):
            return
        self._pos = ordinal
        self._start_pipe(ordinal / self._fps if ordinal > 0 and self._fps else 0.0)

    def _next_frame(self):
        """The next frame in presentation order: parked first, then the pipe.

        Forward reads must go through here rather than ``_read_raw``: a
        previous forward read parks the frames it decoded past its answer, and
        reading the pipe directly would step over them.
        """
        if self._parked:
            return self._parked.popleft()
        return self._read_raw()

    def _can_read_forward(self, target: float) -> bool:
        """Whether the live pipe can reach ``target`` by reading on.

        A scan that walks forward in time (phase 4 steps 0.2 s at a time)
        would otherwise restart ffmpeg for every sample: measured at ~47 ms
        per restart, that turns a few hundred samples into tens of seconds of
        process churn.  Reading on is only valid when the pipe is live and the
        target is strictly after the frame last handed out -- a seek back to
        the frame just delivered must really seek, or a caller that re-reads a
        frame would silently get the next.

        Parked frames do NOT disqualify it: they are the frames the pipe has
        already decoded, in order and all after the last one handed out, so
        reading on through them is exactly right.  Refusing while they were
        pending made every other forward step restart ffmpeg, which is what
        made phase 4's 0.2 s scan cost thousands of process starts.
        """
        if self._proc is None or self._last_pts is None:
            return False
        if not self._fps:
            return False
        return target > self._last_pts + DISPLAY_TIME_TOLERANCE

    def seek_to_pts(self, pts):
        """Position so the next read()/grab() returns the first frame whose
        PTS -- as get_last_pts() reports it -- is at or after ``pts``.

        Returns False at end of stream (and when ``pts`` is past the last
        frame), leaving read()/grab() failing until the next seek.

        The pipe starts a little before ``pts`` and reads forward to it, and a
        seek whose first frame is more than one frame late is retried from
        further back: seeking straight at ``pts`` is not enough, because
        ffmpeg's input seek can discard the frame it was aimed at and emit the
        next one (a 10-bit HEVC clip here loses it), which would hand the
        caller a later frame under the requested timestamp.
        """
        self._read_started = True
        self.seek_past_end = False
        if self._can_read_forward(pts):
            logger.debug("%s: seek_to_pts(%.3f) read forward from %.3f", self.path, pts, self._last_pts)
            return self._read_forward_to_pts(pts)
        start = self._info.start_time
        frame_duration = 1.0 / self._fps if self._fps else 0.0
        preroll = _SEEK_PREROLL_SECONDS
        for attempt in range(_SEEK_MAX_RETRIES + 1):
            self._pos = 0
            seek = max(0.0, (pts - start) - preroll)
            logger.debug("%s: seek_to_pts(%.3f) attempt %d, -ss %.3f (preroll %.3f)",
                         self.path, pts, attempt, seek, preroll)
            self._start_pipe(seek)
            while True:
                got = self._read_raw()
                if got is None:
                    # A seek that decodes nothing up to the end of the stream
                    # has its answer: `pts` is past the last frame.
                    logger.debug("%s: seek_to_pts(%.3f) decoded nothing; past the end", self.path, pts)
                    self._stop_pipe()
                    self.seek_past_end = True
                    return False
                raw, frame_pts = got
                if frame_pts >= pts - DISPLAY_TIME_TOLERANCE:
                    break
            # A frame more than one frame duration after the target means the
            # seek's discard took the frame at the target with it, so retry
            # from further back. Landing within one frame duration is the
            # ordinary answer when `pts` falls between two frames, and at the
            # start of the stream there is nothing earlier to retry from.
            if seek <= 0.0 or frame_pts - pts < frame_duration - DISPLAY_TIME_TOLERANCE:
                logger.debug("%s: seek_to_pts(%.3f) landed on %.3f", self.path, pts, frame_pts)
                self._parked.append((raw, frame_pts))
                self._pos = max(0, int(round((frame_pts - start) * self._fps)))
                return True
            logger.debug("%s: seek_to_pts(%.3f) landed late on %.3f (%.1f frames); "
                         "retrying from further back", self.path, pts, frame_pts,
                         (frame_pts - pts) * self._fps)
            preroll *= 2.0
        logger.warning(
            "seek_to_pts(%r) on %s: every seek, including %d retries from "
            "further back, landed after the target; reporting no frame rather than a "
            "later one", pts, self.path, _SEEK_MAX_RETRIES)
        # Landed late, not past the end: seek_past_end stays False so a caller
        # scanning forward skips this time and carries on instead of deciding
        # the stream ended here.
        self._stop_pipe()
        return False

    def _read_forward_to_pts(self, pts) -> bool:
        """Read the live pipe on until a frame at or after ``pts``."""
        start = self._info.start_time
        while True:
            got = self._next_frame()
            if got is None:
                self._stop_pipe()
                self.seek_past_end = True
                return False
            raw, frame_pts = got
            if frame_pts >= pts - DISPLAY_TIME_TOLERANCE:
                self._parked.appendleft((raw, frame_pts))
                self._pos = max(0, int(round((frame_pts - start) * self._fps)))
                return True

    def seek_to_display_time(self, t):
        """Position so the next read()/grab() returns the frame on screen at
        time ``t``: the last frame whose PTS is at most ``t`` plus
        DISPLAY_TIME_TOLERANCE, or the first frame when ``t`` precedes it.
        Reading on continues with the frame after it.

        This is what a scan that records a decision under a time must read.
        seek_to_pts() instead returns the first frame at or after a time,
        which for a time between two frames is the one not yet shown.

        Returns False when ``t`` is past the last frame (its PTS plus one frame
        at the nominal rate), leaving read()/grab() failing until the next
        seek and ``seek_past_end`` True.
        """
        self._read_started = True
        limit = t + DISPLAY_TIME_TOLERANCE
        start = self._info.start_time
        if self._can_read_forward(t):
            forward = self._read_forward_to_display_time(t)
            if forward is not None:
                return forward
            logger.debug("%s: seek_to_display_time(%.3f) cannot read forward (the frame "
                         "on screen was already consumed by grab()); seeking", self.path, t)
        preroll = _SEEK_PREROLL_SECONDS
        for attempt in range(_SEEK_MAX_RETRIES + 1):
            self.seek_past_end = False
            seek = max(0.0, (limit - start) - preroll)
            logger.debug("%s: seek_to_display_time(%.3f) attempt %d, -ss %.3f (preroll %.3f)",
                         self.path, t, attempt, seek, preroll)
            self._start_pipe(seek)
            on_screen = None
            overshoot = None
            while True:
                got = self._read_raw()
                if got is None:
                    break
                raw, frame_pts = got
                if frame_pts > limit:
                    overshoot = (raw, frame_pts)
                    break
                on_screen = (raw, frame_pts)
            if on_screen is not None:
                # The last frame at or before `t` is on screen.  When the pipe
                # stopped on a frame past `t`, that frame is the very next one
                # after it, so it is parked too rather than decoded again.
                if overshoot is None and limit >= on_screen[1] + 1.0 / self._fps:
                    # Nothing follows the last frame: `t` is past the end.
                    logger.debug("%s: seek_to_display_time(%.3f): nothing after %.3f; past the end",
                                 self.path, t, on_screen[1])
                    self._stop_pipe()
                    self.seek_past_end = True
                    return False
                logger.debug("%s: seek_to_display_time(%.3f) -> frame %.3f%s", self.path, t, on_screen[1],
                             "" if overshoot is None else f", next {overshoot[1]:.3f} parked")
                self._parked.append(on_screen)
                if overshoot is not None:
                    self._parked.append(overshoot)
                self._pos = max(0, int(round((on_screen[1] - start) * self._fps)))
                return True
            if overshoot is not None:
                # The pipe decoded only frames past `t`.  From the start of the
                # stream that means `t` precedes the first frame, which is then
                # on screen; anywhere else the seek landed late, so retry from
                # further back.
                if seek <= 0.0:
                    logger.debug("%s: seek_to_display_time(%.3f) precedes the first frame %.3f",
                                 self.path, t, overshoot[1])
                    self._parked.append(overshoot)
                    self._pos = 0
                    return True
                logger.debug("%s: seek_to_display_time(%.3f) decoded only frames past it "
                             "(first %.3f); retrying from further back", self.path, t, overshoot[1])
                preroll *= 2.0
                continue
            # The pipe decoded nothing at all.
            logger.debug("%s: seek_to_display_time(%.3f) decoded nothing; past the end", self.path, t)
            self._stop_pipe()
            self.seek_past_end = True
            return False
        logger.warning(
            "seek_to_display_time(%r) on %s: every seek, including %d retries from "
            "further back, landed after the target; reporting no frame rather than a "
            "later one", t, self.path, _SEEK_MAX_RETRIES)
        # Landed late, not past the end: seek_past_end stays False so a caller
        # scanning forward skips this time and carries on instead of deciding
        # the stream ended here.
        self._stop_pipe()
        return False

    def _read_forward_to_display_time(self, t):
        """Read the live pipe on to the frame on screen at ``t``.

        Returns True/False like a seek, or None when reading on cannot answer
        and the caller has to seek for real: the pipe's next frame is already
        past ``t``, the frame on screen is the one just handed out, and that
        frame's bytes were not kept (a grab() consumed it).
        """
        limit = t + DISPLAY_TIME_TOLERANCE
        start = self._info.start_time
        on_screen = None
        overshoot = None
        while True:
            got = self._next_frame()
            if got is None:
                # End of the pipe.  Close it so the next forward read is a real
                # seek: left open, every later step would read EOF and re-serve
                # the frame just handed out, forever.
                self._stop_pipe()
                break
            raw, frame_pts = got
            if frame_pts > limit:
                overshoot = (raw, frame_pts)
                break
            on_screen = (raw, frame_pts)
        if on_screen is None and self._last_pts is not None and self._last_pts <= limit:
            # The pipe's next frame is already past `t`, but the frame just
            # handed out is the one on screen -- re-reading it is what the
            # caller is asking for.  A forward read cannot go back, so serve it
            # again from the copy kept for exactly this; reporting "no frame"
            # would tell a caller that a time it can see has nothing on screen.
            if self._last_raw is None:
                self._stop_pipe()
                return None
            on_screen = (self._last_raw, self._last_pts)
        if on_screen is None:
            # The pipe had nothing before `t` left to give: either `t` is past
            # the end, or the pipe ended. Both answer "no frame", and the next
            # seek restarts it.
            self._stop_pipe()
            self.seek_past_end = True
            return False
        if overshoot is None and limit >= on_screen[1] + 1.0 / self._fps:
            # Nothing follows the last frame: `t` is past the end.
            self._stop_pipe()
            self.seek_past_end = True
            return False
        # Push back in reverse: the frame read after `on_screen` goes in first
        # so `on_screen` ends up on top of the deque, in presentation order.
        if overshoot is not None:
            self._parked.appendleft(overshoot)
        self._parked.appendleft(on_screen)
        self._pos = max(0, int(round((on_screen[1] - start) * self._fps)))
        return True

    # -- metadata ----------------------------------------------------------

    def get_last_pts(self) -> float:
        """The PTS (presentation timestamp) of the last frame read, in
        seconds, exactly as ffprobe reports it.  This is the canonical timing
        source for subtitle synchronisation."""
        return self._last_pts

    def get_stream_start_time(self) -> float:
        """The container's ``start_time`` in seconds: time 0 of the output.

        Subtitle times are frame PTS minus this, so every line lands on the
        frame it was read from *as the player paints it*.  A player's zero is
        the container start -- libavformat's format start_time, the earliest of
        all the streams -- which is what mpv rebases to under its default
        ``--rebase-start-time=yes``.  The video stream's start_time is NOT that
        zero: it merely says when the first frame arrives, and subtracting it
        shifts every line earlier by (video start - container start).
        """
        return self._info.start_time if self._info is not None else 0.0


# The one capture implementation; kept as a module-level name because callers
# import it as `Capture`.
Capture = FFmpegCapture

ALLOW_FALLBACK_ENV = "OCR_ALLOW_FFMPEG_FALLBACK"


def capture_backend_name() -> str:
    """Name of the capture backend in use."""
    return "ffmpeg"


def assert_reference_backend() -> None:
    """Raise unless ffmpeg and ffprobe are available.

    There is no second decoder any more, so this only fires on a broken
    install -- but it fires loudly rather than letting the OCR pass fail one
    frame at a time.
    """
    import os
    if FFMPEG_AVAILABLE:
        return
    if os.environ.get(ALLOW_FALLBACK_ENV) == "1":
        return
    raise RuntimeError(
        "ffmpeg and ffprobe are missing, so video cannot be decoded. "
        f"{FFMPEG_ERROR}\n"
        f"To proceed anyway (nothing will decode), set {ALLOW_FALLBACK_ENV}=1."
    )
