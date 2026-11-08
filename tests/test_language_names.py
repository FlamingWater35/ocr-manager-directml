"""The Folder settings language list is every language the OCR engine can read.

`LANGUAGE_NAMES` is static (the settings sheet cannot ask the engine, and a
free-text language must still name a safe output path). This test holds the
list against the engine's own routing tables -- `videocr.ocr_engine.route_for`
and RapidOCR's `route_to_model_key` -- so the two cannot drift apart, e.g.
after a RapidOCR upgrade adds or drops a language, or a new code is added to
the settings without a route. It also checks every listed language names its
output with a well-formed tag (core.project.languages.output_tag).
"""
from __future__ import annotations

import re

import pytest

from app.views.folder_settings_fields import LANGUAGE_NAMES, LANGUAGES, language_label
from core.project.languages import output_tag
from videocr import ocr_engine

TAG = re.compile(r"^[a-z]{2,3}(-([A-Z][a-z]{3}|[A-Z]{2}))?$")      # language, then a script or a region


def test_every_listed_language_has_a_route_of_its_own():
    """No listed language may fall back to the default route: the fallback
    reads `ch`, which would silently read, say, Hindi as Chinese."""
    unrooted = sorted(code for code in LANGUAGE_NAMES if ocr_engine.route_for(code).lang_type == "ch"
                      and code != "ch")
    assert not unrooted, f"listed languages with no route of their own: {unrooted}"


@pytest.mark.parametrize("code", sorted(LANGUAGE_NAMES))
def test_each_listed_language_resolves_to_a_real_model(code):
    """`route_to_model_key` only answers for a language its route supports, so
    this fails if our table maps a code onto a script the model lacks."""
    route = ocr_engine.route_for(code)
    assert ocr_engine.route_model_key(route), code


def test_the_bundled_route_covers_the_languages_it_claims():
    """The bundled PP-OCRv6 model needs no download, so its language set is
    the one that works offline out of the box."""
    bundled = {code for code in LANGUAGE_NAMES if ocr_engine.route_for(code).version == "PP-OCRv6"}
    assert "ch" in bundled and "en" in bundled and "japan" in bundled
    assert not ocr_engine._needs_download(ocr_engine.route_for("ch"))


def test_the_list_is_sorted_by_name_and_every_label_is_unique():
    labels = [language_label(code) for code in LANGUAGES]
    assert sorted(LANGUAGES) == sorted(LANGUAGE_NAMES)
    assert labels == sorted(labels, key=str.casefold)
    assert len(set(labels)) == len(labels)
    assert len({name.casefold() for name in LANGUAGE_NAMES.values()}) == len(LANGUAGE_NAMES)


@pytest.mark.parametrize("code", sorted(LANGUAGE_NAMES))
def test_each_language_names_its_output_with_a_well_formed_tag(code):
    assert TAG.match(output_tag(code)), (code, output_tag(code))


@pytest.mark.parametrize(("code", "tag"), [
    ("ch", "zh"), ("chinese_cht", "zh-Hant"), ("japan", "ja"), ("korean", "ko"),
    ("en", "en"), ("rs_latin", "sr-Latn"),
])
def test_output_tag_literals(code, tag):
    assert output_tag(code) == tag


@pytest.mark.parametrize(("typed", "tag"), [
    ("../x", "x"), ("a/b", "a_b"), ("a\\b", "a_b"), ("Chinese (Simplified)", "Chinese_Simplified"),
    ("..", "zh"), ("/", "zh"), ("", "zh"), ("   ", "zh"), (None, "zh"),
])
def test_a_stored_language_that_is_no_plain_code_still_gives_one_safe_path_component(typed, tag):
    assert output_tag(typed) == tag
