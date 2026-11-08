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

from app.views.folder_settings_fields import LANGUAGE_NAMES, LANGUAGES, MODEL_CHOICES, language_label
from core.project import FolderSettings
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
    download_keys = {key for _route, key in ocr_engine.downloadable_models()}
    assert ocr_engine.route_model_key(ocr_engine.route_for("ch")) not in download_keys


def test_every_tier_is_downloadable_except_the_bundled_default():
    """tiny and medium ship with the app but are fetched by the packager, so
    they are the tiers --download-models has to fetch; small is in the wheel."""
    tiers = ocr_engine.downloadable_tiers()
    assert {tier for _task, tier, _key in tiers} == set(ocr_engine.MODEL_TIERS) - {ocr_engine.DEFAULT_MODEL_TIER}
    assert {task for task, _tier, _key in tiers} == {"det", "rec"}
    assert len(tiers) == 4


def test_the_model_picker_offers_every_tier_the_engine_builds():
    """The sheet's list is static so it can draw without the engine loaded;
    this holds it against the engine's own tiers, in the same order, naming
    the same trade-off, with `small` as the one a new folder starts on."""
    assert [value for value, _text in MODEL_CHOICES] == list(ocr_engine.MODEL_TIERS)
    for value, text in MODEL_CHOICES:
        assert ocr_engine.tier_description(value).split(",")[0].split(" ")[0] in text, (value, text)
    assert FolderSettings().ocr_model == ocr_engine.DEFAULT_MODEL_TIER == "small"


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
