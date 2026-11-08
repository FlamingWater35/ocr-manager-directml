"""How many ffmpeg decoders may run at once, and how many threads each gets.

A folder open used to start up to seventeen ffmpeg processes at the same
moment: the thumbnail job, the filmstrip's frame job, its crop-strip job and
the crop detector's probe batch each fanned out to its own pool, so four
independent pools of one-shot grabs ran side by side. Measured on a 3840x1640
10-bit HEVC file (12 logical cores), that burst held nine cores busy for five
seconds -- which is what "CPU spikes when a video loads" was.

Two levers, measured on that file:

* concurrent decoders -- fifteen one-shot grabs at ten at a time cost 52 s of
  CPU; the same work at four at a time cost 37 s and finished in the same wall
  time. Above about half the core count the extra processes only fight each
  other, and each one still pays a full decoder thread pool to set up.
* threads per decoder -- a one-shot grab decodes a handful of frames, so
  ffmpeg's default "one thread per core" spends its time starting threads.
  ``-threads 2`` cut the batch from 52 s to 45 s of CPU with no wall-time
  change, and a long-lived pipe decode (the OCR pass) went from 3.1 to 2.2
  cores at the same 56 fps with ``-threads 4``.

So the budget is counted in *decoder threads*, not processes: a one-shot grab
takes `grab_threads()` units, a strip capture the same, and the total is
``cores - headroom``. One semaphore is shared by every decode path, so the
burst cannot exceed it however many jobs are in flight.

The long-lived OCR pass's capture is deliberately NOT charged to the budget.
It runs for minutes on the RUN lane while AutoPilot holds every other CPU job
anyway, and charging it would let it starve the previews that are drawn from
the same file. It takes `stream_threads()` and leaves the budget alone.

Nothing here imports Qt. Override the result with ``OCR_MANAGER_DECODE_SLOTS``
(units) or ``OCR_MANAGER_DECODE_THREADS`` (threads per one-shot decode) when a
machine behaves differently; both are read once, at import.
"""
from __future__ import annotations

import contextlib
import os
import threading

SLOTS_ENV = "OCR_MANAGER_DECODE_SLOTS"
THREADS_ENV = "OCR_MANAGER_DECODE_THREADS"

# Cores kept out of the budget so the UI thread, the OCR engine and the rest of
# the app stay responsive while decoding: 2 on a 4-core laptop, 4 on a 32-core
# workstation (capped, because more decoders stop helping before that).
_MAX_HEADROOM = 4


def _cores() -> int:
    return os.cpu_count() or 4


def _env_int(name: str) -> int | None:
    """The override in `name`, or None when unset or unusable."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def grab_threads() -> int:
    """``-threads`` for a one-shot grab: 1 on a small machine, 2 above that.

    A grab decodes a few frames and then exits, so its thread pool is mostly
    startup cost; two threads kept the batch wall time identical while cutting
    its CPU by a seventh, and one thread cost 25% more wall time for a little
    less CPU still."""
    override = _env_int(THREADS_ENV)
    if override is not None:
        return override
    return 1 if _cores() <= 4 else 2


def stream_threads() -> int:
    """``-threads`` for a long-lived pipe decode (the OCR pass, strip samples).

    This one is not startup-bound, so it gets a real pool -- but not ffmpeg's
    default of one thread per core, which measured 3.1 cores for a decode that
    four threads did in 2.2 at the same frame rate."""
    return max(2, min(4, _cores() // 3))


def total_slots() -> int:
    """Decoder threads every one-shot path may use at once."""
    override = _env_int(SLOTS_ENV)
    if override is not None:
        return override
    headroom = min(_MAX_HEADROOM, max(2, _cores() // 3))
    return max(grab_threads(), _cores() - headroom)


class _Budget:
    """A counted permit pool: `units` threads may be in flight at once.

    Written out rather than using `threading.BoundedSemaphore`, which has no
    acquire-many: its `acquire(n)` would take `n` as the *blocking* flag and
    hand out a single permit, and its `release(n)` then raises
    "Semaphore released too many times"."""

    def __init__(self, units: int) -> None:
        self._left = units
        self._units = units
        self._cond = threading.Condition()

    @property
    def units(self) -> int:
        return self._units

    def acquire(self, wanted: int) -> None:
        with self._cond:
            while self._left < wanted:
                self._cond.wait()
            self._left -= wanted

    def release(self, wanted: int) -> None:
        with self._cond:
            self._left = min(self._units, self._left + wanted)
            self._cond.notify_all()

    def free(self) -> int:
        """Permits currently unused; for tests and --self-test."""
        with self._cond:
            return self._left


_BUDGET = _Budget(total_slots())


@contextlib.contextmanager
def slot(units: int | None = None):
    """Hold `units` decoder threads (default: one one-shot grab) while inside.

    Every one-shot decode path wraps its subprocess in this, so four jobs that
    each fan out to a pool of ten still run at most `total_slots() // units`
    decoders together. A unit count larger than the whole budget is clamped:
    a machine tuned down to one slot must still be able to decode."""
    wanted = max(1, min(units if units is not None else grab_threads(), _BUDGET.units))
    _BUDGET.acquire(wanted)
    try:
        yield
    finally:
        _BUDGET.release(wanted)


def describe() -> dict:
    """What --self-test reports, so a slow machine can be diagnosed."""
    return {
        "slots": total_slots(),
        "free": _BUDGET.free(),
        "grab_threads": grab_threads(),
        "stream_threads": stream_threads(),
        "cores": _cores(),
    }
