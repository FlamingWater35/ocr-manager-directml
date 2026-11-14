"""`--download-models` fetches every entry the packager and the installer need.

The command is the one step of the release workflow that touches the network
and the OCR engine at the same time, so a mistake in it is invisible until a
tag is pushed -- which is exactly what happened once: the tier loop referenced
`LangRec` without importing it, and every release died on the tenth of thirteen
models with `NameError: name 'LangRec' is not defined` *after* nine successful
downloads, so the log looked like a network problem.

The tests here drive the real `download_models` against a fake `RapidOCR` that
records the parameters instead of fetching anything. That keeps them offline and
fast while still executing the whole loop -- every name it resolves, every
parameter it builds, and which half of a tier entry it asks for. A test that
only checked `downloadable_tiers()` (as `test_language_names.py` does) could not
have caught the missing import, because the name is only touched at fetch time.
"""
from __future__ import annotations

import pytest

import rapidocr

from videocr import ocr_engine


class FakeRapidOCR:
    """Stands in for `rapidocr.RapidOCR`, recording what each call asked for.

    `download_models` builds the engine, then asks for one half of it
    (`_load_det_model` or `_load_rec_model`); the calls are recorded rather
    than performed because the real ones hit modelscope.cn.
    """

    calls: list[tuple[dict, str]] = []

    def __init__(self, params=None):
        self.params = dict(params or {})
        self.calls.append((self.params, "build"))

    def _load_det_model(self):
        self.calls.append((self.params, "det"))

    def _load_rec_model(self):
        self.calls.append((self.params, "rec"))


@pytest.fixture
def fake_engine(monkeypatch):
    FakeRapidOCR.calls = []
    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)
    return FakeRapidOCR


def test_every_model_is_fetched_and_the_run_reports_all_of_them(fake_engine, tmp_path):
    """The count the progress line shows is `len(models) + len(tiers)`, so a
    loop that stops early -- or one that raises -- is visible as a short list."""
    seen: list[str] = []

    fetched = ocr_engine.download_models(
        tmp_path, progress=lambda index, total, key: seen.append(f"{index + 1}/{total} {key}"))

    expected = [key for _route, key in ocr_engine.downloadable_models()]
    expected += [key for _task, _tier, key in ocr_engine.downloadable_tiers()]
    assert fetched == expected
    assert seen == [f"{index + 1}/{len(expected)} {key}" for index, key in enumerate(expected)]


def test_a_tier_entry_asks_only_for_the_half_it_names(fake_engine, tmp_path):
    """Building both halves would download the other one twice: four tier
    entries are four files, not eight."""
    ocr_engine.download_models(tmp_path)

    half: dict[tuple[str, str], int] = {}
    for params, call in fake_engine.calls:
        if call in ("det", "rec"):
            key = (params["Det.model_type"].value, call)
            half[key] = half.get(key, 0) + 1
    # The nine language routes reuse the bundled `small` detector and only ask
    # for the recogniser; each of the two unbundled sizes asks for one of each.
    assert half == {("small", "rec"): len(ocr_engine.downloadable_models()),
                    ("tiny", "det"): 1, ("tiny", "rec"): 1,
                    ("medium", "det"): 1, ("medium", "rec"): 1}


def test_the_parameters_are_enums_and_the_language_is_a_plain_string(fake_engine, tmp_path):
    """`ParseParams.update_batch` rejects a string for `model_type` and accepts
    one for `lang_type` (it converts through `LangRec` itself), which is why
    the tier entries pass `"ch"` rather than reaching for the enum."""
    ocr_engine.download_models(tmp_path)

    for params, _call in fake_engine.calls:
        assert params["Global.log_level"] == "warning"
        assert params["Global.model_root_dir"] == str(tmp_path)
        assert params["EngineConfig.onnxruntime.use_dml"] is False
        assert params["Det.model_type"].__class__.__name__ == "ModelType"
        assert params["Det.ocr_version"].__class__.__name__ == "OCRVersion"
        assert isinstance(params["Rec.lang_type"], str)


def test_cancelling_before_an_entry_stops_without_fetching_it(fake_engine, tmp_path):
    """The CLI cancels through a `threading.Event`; the loop checks it before
    each model so Ctrl-C does not have to wait for a 20 MB download."""
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(RuntimeError, match="cancelled"):
        ocr_engine.download_models(tmp_path, cancel=cancel)

    assert len([call for _params, call in fake_engine.calls if call in ("det", "rec")]) < 4
