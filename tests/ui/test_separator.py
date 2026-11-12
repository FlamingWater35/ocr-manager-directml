"""One separator, everywhere: "|" between values, never the middle dot.

The user's screenshot was the crop canvas's two chips:
    3840 × 1640 · t 10:48.68        Crop 576, 1434 · 2688 × 141
At the app's scope size the middle dot reads as a full stop in mid-sentence,
not as a divider between two values, so every value line now uses a light
vertical rule. These tests keep one separator from creeping back in as a
second style, and keep the ones that ARE dividers joined by the same glyph.
"""
from __future__ import annotations

import os
import re

import pytest

from app import recents, run_snapshot, state_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOT = "\u00b7"


# --------------------------------------------------------------------------
# The text builders
# --------------------------------------------------------------------------

def test_crop_text_divides_with_a_pipe():
    class Crop:
        x, y, width, height = 288, 784, 1344, 55

    assert state_text.crop_text(Crop()) == "288, 784 | 1344 × 55"


def test_media_text_divides_with_a_pipe():
    class Media:
        width, height, duration, fps = 1920, 888, 1628.0, 25.0

    assert state_text.media_text(Media()) == "1920×888 | 27:08 | 25 fps"


def test_flag_reasons_join_with_a_pipe():
    from core.detect import brightness as brightness

    flagged = "+".join([brightness.FLAG_NARROW_PLATEAU, brightness.FLAG_DIM_TEXT])
    text = state_text.brightness_flag_text(flagged)
    assert " | " in text
    assert DOT not in text


def test_run_status_divides_with_a_pipe():
    text = run_snapshot.run_status_text(
        run_snapshot.RunSnapshot(files=(), started_at=0.0, parallel=1), 0.0)
    assert text.startswith("| ")
    assert " | " in text
    assert DOT not in text


def test_recent_jobs_divides_with_a_pipe():
    from app.views.activity import recent_text

    text = recent_text([("crop", "ep01.mkv", 0.0)], 11.0)
    assert text.startswith("| ")
    assert DOT not in text


def test_a_recents_row_divides_with_a_pipe():
    entry = recents.RecentEntry(path="D:/x", kind=recents.FOLDER, opened=0.0, count=12)
    text = recents.detail(entry, 7200.0)
    assert " | " in text
    assert DOT not in text


# --------------------------------------------------------------------------
# The source: no view may reintroduce the dot in a drawn string
# --------------------------------------------------------------------------

def _drawn_strings(path: str) -> list[tuple[int, str]]:
    """String literals that are not docstrings -- what the window can show."""
    import ast
    import io
    import tokenize

    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    spans = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                value = body[0].value
                if isinstance(value.value, str):
                    spans.add((value.lineno, value.col_offset))
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in (tokenize.STRING, tokenize.FSTRING_MIDDLE):
            continue
        if (token.start[0], token.start[1]) in spans:
            continue
        if DOT in token.string:
            found.append((token.start[0], token.string))
    return found


APP_MODULES = [
    "app/state_text.py", "app/run_snapshot.py", "app/recents.py", "app/activity.py",
    "app/main_window.py", "app/views/crop_view.py", "app/views/brightness_view.py",
    "app/views/activity.py", "app/views/inspector_sections.py", "app/views/ranges_view.py",
    "app/views/queue.py", "app/views/topbar.py", "app/views/folder_settings.py",
    "app/views/run_view.py", "app/views/episode/prepare.py", "app/views/episode/done.py",
]


@pytest.mark.parametrize("relative", APP_MODULES)
def test_no_drawn_string_still_uses_the_middle_dot(relative):
    """Comments and docstrings are prose and may keep the dot; a literal the
    window draws may not, or two separators would be in use at once."""
    path = os.path.join(ROOT, relative)
    offenders = _drawn_strings(path)
    assert not offenders, "\n".join(f"{relative}:{row}: {text!r}" for row, text in offenders)


def test_the_separator_is_one_glyph_everywhere():
    """Every joined line uses " | ", not a mix of the dot and the pipe: a
    reader who learns the rule from one line must find it on the next."""
    sample = [
        state_text.crop_text(type("C", (), {"x": 1, "y": 2, "width": 3, "height": 4})()),
        run_snapshot.run_status_text(
            run_snapshot.RunSnapshot(files=(), started_at=0.0, parallel=1), 0.0),
    ]
    for text in sample:
        assert DOT not in text
        assert " | " in text or text.startswith("| ")
