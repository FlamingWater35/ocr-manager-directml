"""Dialogue progress must follow the OCR work, not the frame reads.

`Video.run_ocr` reads frames on a producer thread and OCRs them in batches on
the consumer thread. Counting only the producer's reads makes dialogue report
100 % while the last batches are still running: measured on
test_media/test_video.mkv's 30 s proof window the reads finished at 32.8 s and
the OCR work at 62.3 s, so the bar sat at 100 % for half a minute -- which on a
progress bar is indistinguishable from a hang.

The contract is a split, and it is checked two ways:

* **Structurally**: the producer counts ONLY the frames it drops, and the
  consumer counts the frames it receives. Together they count every frame of
  the window exactly once, and the count keeps moving while batches run.
* **Behaviourally** (the real producer/consumer loop over a synthetic stream,
  no media and no GPU): the counter reaches exactly 100 %, never overshoots,
  and never goes backwards, and `update()` is atomic under both threads.

A synthetic run cannot reproduce the 34 s stall itself -- a 64 px fixture never
makes the frame queue bind the way real 4K frames do, so both counting schemes
finish in the same order there. That is why the structural half carries the
regression: it is the part that fails if the split is undone.
"""
import ast
import inspect
import threading

import numpy as np
import pytest

from videocr import engine_registry, video as video_mod
from videocr.progress import ProgressTracker
from videocr.video import Video

FPS = 25.0
FRAME_SIZE = 64
FRAMES_PER_SUBTITLE = 60
SUBTITLES = 40
TOTAL_FRAMES = SUBTITLES * FRAMES_PER_SUBTITLE

BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]


# --------------------------------------------------------------------------
# The split, read from the source
# --------------------------------------------------------------------------

def _dedent(source: str) -> str:
    lines = source.splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    strip = min(indents) if indents else 0
    return "\n".join(line[strip:] if line.strip() else line for line in lines)


def _producer_source() -> str:
    return _dedent(inspect.getsource(Video._frame_producer))


def _consumer_source() -> str:
    return _dedent(inspect.getsource(Video._run_ocr_with_engine))


def _branch(source: str, opener: str, closers: tuple[str, ...]) -> str:
    """The body of `opener`'s branch, up to whichever `closers` entry comes
    first. Used instead of an AST walk because the branches are a flat chain of
    `elif` arms, which string boundaries express more directly."""
    body = source.split(opener)[1]
    cuts = [body.index(closer) for closer in closers if closer in body]
    return body[:min(cuts)] if cuts else body


def _producer_tree():
    return ast.parse(_producer_source())


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _body_lists(node, blocks=()):
    """Every statement list in the tree, innermost-last, as (list, parents)."""
    for field, value in ast.iter_fields(node):
        if isinstance(value, list) and value and all(isinstance(i, ast.stmt) for i in value):
            yield value, blocks
            for statement in value:
                yield from _body_lists(statement, blocks + (value,))
        elif isinstance(value, ast.AST):
            yield from _body_lists(value, blocks)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield from _body_lists(item, blocks)


def _is_call(statement, dotted: str) -> bool:
    return (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
            and _dotted(statement.value.func) == dotted)


def _queues_directly(block) -> set[str]:
    """The message kinds queued by a DIRECT statement of this body list."""
    kinds = set()
    for statement in block:
        if not _is_call(statement, "queue.put"):
            continue
        args = statement.value.args
        if args and isinstance(args[0], ast.Tuple) and args[0].elts:
            head = args[0].elts[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                kinds.add(head.value)
    return kinds


def _counts_directly(block) -> bool:
    """Whether a DIRECT statement of this body list counts a frame -- either a
    bare call, or the `if progress is not None: progress.update(1)` wrapper the
    code uses. Nested branches are separate body lists and are judged on their
    own."""
    for statement in block:
        if _is_call(statement, "progress.update"):
            return True
        if isinstance(statement, ast.If) and not statement.orelse:
            if any(_is_call(child, "progress.update") for child in statement.body):
                return True
    return False


def test_the_producer_does_not_count_a_frame_it_queues():
    """The bug, pinned. A queued frame is counted by the CONSUMER; if the
    producer counts it too, dialogue reaches 100 % at the last READ and the bar
    freezes for the whole batch tail (34 s measured on the 4K window).

    Judged per BRANCH, on direct statements only: the rule is that no single
    body list both queues a frame and counts one. A `progress.update` in the
    sibling `elif` (the frames that are dropped) is correct, and a nested arm's
    count belongs to that arm."""
    tree = _producer_tree()
    offenders = []
    for block, _parents in _body_lists(tree.body[0]):
        queued = _queues_directly(block) & {"frame", "extend"}
        if queued and _counts_directly(block):
            offenders.append(sorted(queued))
    assert not offenders, (
        f"the producer counts frames it queues ({offenders!r}); the consumer counts "
        "those too, so dialogue finishes early and the bar freezes while the OCR "
        "batches drain")


def test_the_producer_counts_the_frames_it_drops():
    """The other half: frames the producer decides not to send (no text yet,
    text just disappeared) are never seen by the consumer, so the producer must
    count them or the total is unreachable and dialogue never hits 100 %."""
    source = _producer_source()
    assert "elif progress is not None:" in source, (
        "the producer no longer counts the frames it drops, so the frame total "
        "is unreachable and dialogue can never report 100 %")
    assert source.count("progress.update(1)") >= 3


def test_the_consumer_counts_every_frame_it_receives():
    source = _consumer_source()
    for kind in ("frame", "extend"):
        body = _branch(source, f'elif msg_type == "{kind}":',
                       ('elif msg_type', "finally:"))
        assert "progress.update(1)" in body, (
            f"the consumer no longer counts a {kind!r} message, so the frames it "
            "processes are never reported and the bar stalls")


# --------------------------------------------------------------------------
# The counter itself
# --------------------------------------------------------------------------

def _reports(**kwargs):
    seen = []
    tracker = ProgressTracker(progress_callback=lambda name, pct: seen.append((name, pct)), **kwargs)
    return tracker, seen


def test_dialogue_progress_reaches_exactly_100_percent():
    tracker, seen = _reports()
    tracker.set_phase("dialogue", 10)
    for _ in range(10):
        tracker.update(1)
    assert seen[0] == ("Extracting dialogue", 0)
    assert seen[-1] == ("Extracting dialogue", 100)
    assert [pct for _, pct in seen] == sorted(pct for _, pct in seen)


def test_dialogue_progress_cannot_exceed_100_percent():
    """Both threads can briefly over-count while a frame is in flight; the
    percentage must clamp rather than overshoot the bar."""
    tracker, seen = _reports()
    tracker.set_phase("dialogue", 4)
    for _ in range(9):
        tracker.update(1)
    assert max(pct for _, pct in seen) == 100


def test_update_is_atomic_under_two_threads():
    """Both reporter threads call update() for every frame. A non-atomic
    counter loses increments, and the bar then never reaches 100 %."""
    total = 4000
    tracker, seen = _reports()
    tracker.set_phase("dialogue", total)

    def count(n):
        for _ in range(n):
            tracker.update(1)

    threads = [threading.Thread(target=count, args=(total // 2,)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(pct for _, pct in seen) == 100
    assert seen[-1] == ("Extracting dialogue", 100)


def test_a_zero_unit_phase_still_advances():
    tracker, seen = _reports()
    tracker.set_phase("dialogue", 0)
    tracker.update(1)
    assert seen[-1] == ("Extracting dialogue", 100)


def test_labels_are_weighted_across_their_phases():
    tracker, seen = _reports(include_dialogue=False)
    weights = tracker._label_weights
    tracker.set_phase("label_p1", 10)
    for _ in range(10):
        tracker.update(1)
    assert seen[-1] == ("Extracting labels", weights["label_p1"])
    tracker.set_phase("label_p1_5", 2)
    tracker.update(2)
    assert seen[-1] == ("Extracting labels", weights["label_p1"] + weights["label_p1_5"])


# --------------------------------------------------------------------------
# The real loop, end to end
# --------------------------------------------------------------------------

def _frame(value: int, index: int) -> np.ndarray:
    frame = np.full((FRAME_SIZE, FRAME_SIZE, 3), value, dtype=np.uint8)
    frame[0, 0, 0] = index // 256
    frame[0, 0, 1] = index % 256
    return frame


def _stream() -> list:
    frames = []
    for index in range(TOTAL_FRAMES):
        block = index // FRAMES_PER_SUBTITLE
        frames.append(_frame(10 if block % 2 == 0 else 240, index))
    return frames


class StubOCR:
    def __init__(self):
        self.calls = 0

    def ocr(self, frames):
        self.calls += 1
        return [[[BOX, ("文", 0.9)]] for _ in frames]

    def predict(self, frames):
        return self.ocr(frames)


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_stream_start_time(self):
        return 0.0

    def set(self, prop, value):
        return True

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        frame = self.frames[self.i]
        self.i += 1
        return True, frame

    def get_last_pts(self):
        return (self.i - 1) / FPS


def _run(monkeypatch):
    frames = _stream()
    monkeypatch.setattr(video_mod.utils, "needs_conversion", lambda: False)
    monkeypatch.setattr(video_mod, "Capture", lambda path, **kwargs: FakeCapture(frames))
    ocr = StubOCR()
    monkeypatch.setattr(video_mod.utils, "create_ocr_engine", lambda *a, **k: ocr)
    engine_registry.reset_registry()

    reports = []
    tracker = ProgressTracker(progress_callback=lambda name, pct: reports.append((name, pct)))
    v = Video.__new__(Video)
    v.path = "synthetic"
    v.det_model_dir = v.rec_model_dir = None
    v.fps = FPS
    v.num_frames = TOTAL_FRAMES
    v.height = v.width = FRAME_SIZE
    v.run_ocr(
        use_gpu=False, lang="ch", time_start="0:00", time_end="",
        conf_threshold=95, use_fullframe=True, brightness_threshold=0,
        similar_image_threshold=0.3, similar_pixel_threshold=25,
        frames_to_skip=0, crop_x=None, crop_y=None,
        crop_width=None, crop_height=None, progress=tracker,
    )
    return reports


def test_a_real_run_reports_dialogue_to_exactly_100_percent(monkeypatch):
    """The split has to add up: every frame is counted once by one of the two
    threads, so a run over N frames ends at 100 % and never overshoots."""
    reports = _run(monkeypatch)
    dialogue = [pct for name, pct in reports if name == "Extracting dialogue"]
    assert dialogue, "the run reported no dialogue progress at all"
    assert dialogue[0] == 0
    assert dialogue[-1] == 100, f"dialogue ended at {dialogue[-1]} %, not 100 %"
    assert dialogue == sorted(dialogue), "the bar went backwards"
