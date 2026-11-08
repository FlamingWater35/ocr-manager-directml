"""The decode budget: how many ffmpeg decoders may run at once.

These pin the numbers a folder open depends on. The bug they exist for: four
CPU-lane jobs each fanned out to their own pool of one-shot grabs, so opening
one 4K file started seventeen ffmpeg processes and held nine cores busy for
five seconds -- reported as "CPU spikes when loading a video".
"""
from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import pytest

from core import decode_budget


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    """Every test sees a fresh budget: `_BUDGET` is module state."""
    budget = decode_budget._Budget(decode_budget.total_slots())
    monkeypatch.setattr(decode_budget, "_BUDGET", budget)
    return budget


def test_a_grab_asks_for_a_couple_of_threads_not_one_per_core():
    threads = decode_budget.grab_threads()
    assert 1 <= threads <= 2
    assert decode_budget.describe()["cores"] >= 1


def test_a_long_lived_stream_gets_a_smaller_pool_than_ffmpeg_defaults_to(monkeypatch):
    monkeypatch.setattr(decode_budget, "_cores", lambda: 16)
    assert decode_budget.stream_threads() == 4          # not 16
    monkeypatch.setattr(decode_budget, "_cores", lambda: 2)
    assert decode_budget.stream_threads() == 2          # never below two


def test_the_budget_leaves_cores_for_the_ui_and_the_ocr_engine(monkeypatch):
    monkeypatch.setattr(decode_budget, "_cores", lambda: 12)
    assert decode_budget.total_slots() == 8             # 12 - min(4, 4)
    monkeypatch.setattr(decode_budget, "_cores", lambda: 4)
    assert decode_budget.total_slots() == 2             # 4 - 2
    monkeypatch.setattr(decode_budget, "_cores", lambda: 64)
    assert decode_budget.total_slots() == 60            # headroom capped at 4


def test_the_budget_is_never_smaller_than_one_grab(monkeypatch):
    monkeypatch.setattr(decode_budget, "_cores", lambda: 1)
    assert decode_budget.total_slots() >= decode_budget.grab_threads()


def test_a_slot_holds_its_threads_and_gives_them_back():
    total = decode_budget.total_slots()
    assert decode_budget.describe()["free"] == total
    with decode_budget.slot():
        assert decode_budget.describe()["free"] == total - decode_budget.grab_threads()
    assert decode_budget.describe()["free"] == total


def test_a_slot_holds_its_threads_again_after_an_exception():
    total = decode_budget.total_slots()
    with pytest.raises(RuntimeError):
        with decode_budget.slot():
            raise RuntimeError("decode blew up")
    assert decode_budget.describe()["free"] == total


def test_a_request_bigger_than_the_whole_budget_is_clamped(monkeypatch):
    monkeypatch.setattr(decode_budget, "_BUDGET", decode_budget._Budget(2))
    with decode_budget.slot(99):
        assert decode_budget._BUDGET.free() == 0
    assert decode_budget._BUDGET.free() == 2


def test_only_the_budget_worth_of_decoders_run_at_once():
    """The regression: a fan-out of twenty must not become twenty ffmpeg."""
    total = decode_budget.total_slots()
    peak = 0
    live = 0
    lock = threading.Lock()
    release = threading.Event()

    def decode():
        nonlocal peak, live
        with decode_budget.slot():
            with lock:
                live += 1
                peak = max(peak, live)
            release.wait(0.01)
            with lock:
                live -= 1

    threads = [threading.Thread(target=decode) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert peak <= total // decode_budget.grab_threads()
    assert peak >= 1


def test_a_blocked_decoder_waits_rather_than_failing(monkeypatch):
    monkeypatch.setattr(decode_budget, "_BUDGET", decode_budget._Budget(1))
    held = threading.Event()
    done = threading.Event()

    def first():
        with decode_budget.slot(1):
            held.set()
            done.wait(5)

    def second():
        held.wait(5)
        with decode_budget.slot(1):
            done.set()

    a = threading.Thread(target=first)
    b = threading.Thread(target=second)
    a.start()
    b.start()
    assert held.wait(5)
    time.sleep(0.05)
    assert not done.is_set()            # b is waiting on the budget, not on an error
    done.set()
    a.join(5)
    b.join(5)
    assert decode_budget._BUDGET.free() == 1


def test_the_env_overrides_are_read(monkeypatch):
    monkeypatch.setenv(decode_budget.SLOTS_ENV, "3")
    monkeypatch.setenv(decode_budget.THREADS_ENV, "1")
    assert decode_budget.total_slots() == 3
    assert decode_budget.grab_threads() == 1


@pytest.mark.parametrize("raw", ["", "0", "-2", "many", "2.5"])
def test_an_unusable_override_falls_back_to_the_measured_default(monkeypatch, raw):
    monkeypatch.setenv(decode_budget.SLOTS_ENV, raw)
    assert decode_budget.total_slots() == decode_budget._cores() - min(
        decode_budget._MAX_HEADROOM, max(2, decode_budget._cores() // 3))


def _calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


ONE_SHOT = [
    Path("core/detect/crop.py"),
]


def test_every_one_shot_grab_charges_the_budget_and_caps_its_threads():
    """A grab that skips either one is how the burst came back before."""
    for path in ONE_SHOT:
        source = path.read_text(encoding="utf-8")
        assert "decode_budget.slot()" in source, f"{path} does not charge the decode budget"
        assert "decode_budget.grab_threads()" in source, f"{path} does not cap its threads"


def test_the_streaming_capture_caps_its_threads_before_the_input():
    """`-threads` after -i only caps the muxer: measured no CPU change at all."""
    source = Path("videocr/ffmpeg_capture.py").read_text(encoding="utf-8")
    assert "decode_budget.stream_threads()" in source
    threads_at = source.index('"-threads", str(decode_budget.stream_threads())')
    input_at = source.index('"-i", self.path')
    assert threads_at < input_at


def test_the_strip_sampler_charges_the_budget():
    source = Path("core/detect/ocr_view.py").read_text(encoding="utf-8")
    assert "decode_budget.slot()" in source
