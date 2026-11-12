"""Stop must end a run even when the frame producer is blocked.

The defect: `_frame_producer` put frames with a plain `queue.put()`. When the
frame queue was full and the consumer had already left on a cancel, that put
blocked forever -- the consumer was in `producer.join()`, so the worker never
returned, the runner never emitted a terminal event and the window sat on
"Stopping..." with no way out.

`_put_cancellable` is the fix: it re-checks the cancel event every
`QUEUE_POLL_SECONDS` and gives up instead of blocking. These tests pin both
halves -- the helper's own contract, and the fact that the producer uses it
for every message it sends.
"""
from __future__ import annotations

import ast
import threading
import time
from queue import Queue

import pytest

import videocr.video as video_module
from videocr.video import _put_cancellable


# --------------------------------------------------------------------------
# The helper's contract
# --------------------------------------------------------------------------

def test_a_put_into_a_roomy_queue_returns_true():
    queue: Queue = Queue()
    assert _put_cancellable(queue, "frame", None) is True
    assert queue.get_nowait() == "frame"


def test_a_put_gives_up_once_the_run_is_cancelled():
    """The whole point: a full queue plus a cancel must not block."""
    queue: Queue = Queue(maxsize=1)
    queue.put("already here")                    # now full
    cancel = threading.Event()
    cancel.set()                                 # cancelled before the put
    started = time.monotonic()
    assert _put_cancellable(queue, "blocked frame", cancel) is False
    assert time.monotonic() - started < 1.0, "it returned without waiting for room"


def test_a_put_already_blocked_notices_a_later_cancel():
    """The real deadlock: the producer is inside the put when Stop arrives."""
    queue: Queue = Queue(maxsize=1)
    queue.put("already here")
    cancel = threading.Event()
    result: list[bool] = []

    def producer() -> None:
        result.append(_put_cancellable(queue, "blocked frame", cancel))

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    time.sleep(video_module.QUEUE_POLL_SECONDS * 2)   # let it block
    assert thread.is_alive(), "the producer should be waiting for room"
    cancel.set()
    thread.join(timeout=video_module.QUEUE_POLL_SECONDS * 4)
    assert not thread.is_alive(), "the producer did not notice the cancel"
    assert result == [False]


def test_the_poll_interval_is_short_enough_to_feel_like_stop():
    """A stop that takes a minute to notice is a stop that looks hung. The
    poll is also the consumer's wait, so it cannot be a long one."""
    assert 0.05 <= video_module.QUEUE_POLL_SECONDS <= 1.0


# --------------------------------------------------------------------------
# The producer uses it everywhere
# --------------------------------------------------------------------------

def _producer_source() -> str:
    path = video_module.__file__
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _producer_body() -> str:
    source = _producer_source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_frame_producer":
            return ast.get_source_segment(source, node) or ""
    pytest.fail("_frame_producer is gone; this test needs rewriting")


def test_the_producer_never_puts_without_the_cancel_check():
    """A single plain `queue.put(...)` reintroduces the deadlock for that
    message, so the check is on the source: every put in the producer is the
    cancellable one."""
    body = _producer_body()
    assert "queue.put(" not in body, (
        "the producer must send through _put_cancellable(), not queue.put()")
    assert body.count("_put_cancellable(") >= 4, (
        "expected the frame, candidate, extend and done sends to all be cancellable")


def test_the_done_message_is_cancellable_too():
    """`done` is the last message; the consumer may already have left, and a
    full queue would block the producer on the one put nobody drains."""
    body = _producer_body()
    done_line = next((line for line in body.splitlines() if '("done"' in line), "")
    assert "_put_cancellable(" in done_line, done_line


def test_the_consumer_waits_with_the_same_poll():
    """The consumer's `get` timeout is what lets it re-check the cancel event;
    sharing the constant keeps the two halves in step."""
    source = _producer_source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_run_ocr_with_engine":
            body = ast.get_source_segment(source, node) or ""
            assert "QUEUE_POLL_SECONDS" in body
            return
    pytest.fail("_run_ocr_with_engine is gone; this test needs rewriting")
