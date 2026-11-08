"""The OCR engine: RapidOCR on ONNX Runtime, DirectML when it is there.

PaddleOCR is gone. This module is the only place that knows what an engine is;
everything downstream (`videocr/video.py`, `videocr/models.py`,
`videocr/label_scanner.py`, `core/detect/*`) keeps speaking the PaddleOCR 3.x
result shape -- a dict with `rec_texts` / `rec_scores` / `rec_polys` for a full
read and `dt_polys` / `dt_scores` for detection -- so the OCR pipeline and its
tests did not have to change when the engine did.

Two decisions worth knowing:

**DirectML, not CUDA.** `onnxruntime-directml` ships the `DmlExecutionProvider`,
which runs on any DX12 GPU (NVIDIA, AMD, Intel) and falls back to the CPU
without a driver. There is no CUDA toolkit, no `nvidia-smi`, no per-driver
build to pick: one wheel, one code path.

**The engine never silently runs on the CPU.** RapidOCR's provider setup
(`rapidocr/inference_engine/onnxruntime/provider_config.py`) only *warns* when
DirectML is unavailable and quietly falls back to `CPUExecutionProvider`, which
turns a 20x speedup into a mystery. So `create_*` asserts the provider list the
sessions actually got and logs what it settled on; `provider_names()` exposes
it for the installer's verify step.

Language routing: RapidOCR's PP-OCRv6 is a single multilingual model bundled in
the wheel (no download) that covers 52 language keys; the remaining PaddleOCR
languages route to PP-OCRv5 script models (Cyrillic, Devanagari, Arabic, Thai,
Greek, Korean, Tamil, Telugu) or, for Georgian, PP-OCRv4. `LANG_ROUTES` is that
table, and it is the only thing the installer has to prefetch: detection always
uses the bundled PP-OCRv6 detector, which is script-agnostic, so only
recognition models are ever downloaded.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# --- models ---------------------------------------------------------------

# PP-OCRv6 ships inside the rapidocr wheel (`rapidocr/models/`): the detector,
# the multilingual recogniser and the orientation classifier, 31.7 MB in all.
BUNDLED_DET = ("PP-OCRv6", "small")
BUNDLED_CLS_FILE = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
BUNDLED_CLS = ("PP-OCRv4", "mobile")
DET_MODEL = BUNDLED_DET

# Recognition routes: (ocr_version, model_type). Only the models named here are
# ever downloaded, and only the first time a language that needs one is used.
V6 = ("PP-OCRv6", "small")
V5 = ("PP-OCRv5", "mobile")
V4 = ("PP-OCRv4", "mobile")

# --- model sizes ----------------------------------------------------------
#
# PP-OCRv6 comes in three sizes and the user picks one per folder. They are the
# same architecture at three widths, so the choice trades speed against
# accuracy and nothing else in the pipeline changes.
#
# `small` is the default because it is what the wheel carries and what every
# measurement in this codebase was taken with. `tiny` is ~5x smaller than
# `small` and noticeably quicker on a weak GPU; `medium` is ~6x larger and the
# most accurate, at roughly 3x the inference time of `small`.
#
# Only `small` is in the wheel. `tiny` and `medium` are fetched once into the
# model cache (`<data>/models`) the first time a folder asks for them, and the
# Windows bundle ships all three so a released build never downloads at all.
MODEL_TIERS = ("tiny", "small", "medium")
DEFAULT_MODEL_TIER = "small"

# Where each tier's models come from. RapidOCR's own resolver knows these URLs
# (its `default_models.yaml`), so nothing here has to duplicate them: a tier
# that is not bundled is simply left to RapidOCR to fetch into the cache.
_TIER_DESCRIPTIONS = {
    "tiny": "fastest, least accurate",
    "small": "balanced (default)",
    "medium": "slowest, most accurate",
}


def tier_description(tier: str) -> str:
    """The one-line explanation the settings sheet shows under the picker."""
    return _TIER_DESCRIPTIONS.get(tier, "")


def normalise_tier(tier: str | None) -> str:
    """A tier name that is safe to build an engine with.

    An unknown or missing value falls back to the default rather than failing:
    a project file written by a newer build, or a hand-edited one, should not
    stop a run.
    """
    if tier in MODEL_TIERS:
        return tier
    if tier:
        logger.warning("unknown OCR model size %r; using %s", tier, DEFAULT_MODEL_TIER)
    return DEFAULT_MODEL_TIER


@dataclass(frozen=True)
class Route:
    """Which recognition model reads a language, and the `lang_type` that
    resolves to it (`rapidocr.utils.model_resolver.route_to_model_key`)."""

    version: str
    model_type: str
    lang_type: str


# Languages PP-OCRv6 reads natively: one unified model for all of them.
_V6_LANGS = frozenset({
    "af", "az", "bs", "ca", "ch", "chinese_cht", "cs", "cy", "da", "de", "en",
    "es", "et", "eu", "fi", "fr", "french", "ga", "german", "gl", "hr", "hu",
    "id", "is", "it", "japan", "ku", "la", "lb", "lt", "lv", "mi", "ms", "mt",
    "nl", "no", "oc", "pl", "pt", "qu", "rm", "ro", "rs_latin", "sk", "sl",
    "sq", "sv", "sw", "tl", "tr", "uz", "vi",
})

# PaddleOCR languages whose script PP-OCRv5 reads as one model per script.
# (RapidOCR's own `lang_type` keys: cyrillic, devanagari, arabic, th, el,
# korean, ta, te -- `route_to_model_key` rejects a bare ISO code it does not
# list, so the mapping has to happen here.)
_CYRILLIC = frozenset({
    "ru", "uk", "be", "bg", "mk", "mn", "sr", "rs_cyrillic", "kk", "ky", "tg",
    "tt", "ba", "cv", "abq", "ady", "ava", "che", "dar", "inh", "kbd", "lbe",
    "lez", "os", "tab", "xal", "sah", "tyv", "udm", "kv", "mhr", "kaa", "bua",
})
_DEVANAGARI = frozenset({
    "hi", "mr", "ne", "sa", "bh", "mai", "bho", "ang", "mah", "new", "pi",
    "gom", "sck", "bgc",
})
_ARABIC = frozenset({"ar", "fa", "ur", "ug", "ps", "sd", "bal"})

# Languages with a dedicated PP-OCRv5 model of their own.
_V5_OWN = frozenset({"el", "korean", "ta", "te", "th"})

# Georgian is the one language RapidOCR only reads with PP-OCRv4.
_V4_OWN = {"ka": "ka"}

# PaddleOCR's Moldovan is Latin-script Romanian; PP-OCRv6 has "ro".
_ALIASES = {"mo": "ro"}


def route_for(lang: str) -> Route:
    """The recognition route for a PaddleOCR language value.

    Unknown values fall back to the bundled multilingual model with `ch`,
    which is what RapidOCR itself defaults to -- a language the app does not
    list (the field is free text) still gets read rather than raising.
    """
    code = str(lang or "").strip()
    code = _ALIASES.get(code, code)

    if code in _V6_LANGS:
        return Route(*V6, code)
    if code in _V5_OWN:
        return Route(*V5, code)
    if code in _V4_OWN:
        return Route(*V4, code)
    if code in _CYRILLIC:
        return Route(*V5, "cyrillic")
    if code in _DEVANAGARI:
        return Route(*V5, "devanagari")
    if code in _ARABIC:
        return Route(*V5, "arabic")

    logger.debug("%r is not a language RapidOCR knows; reading it as %r", lang, "ch")
    return Route(*V6, "ch")


def prefetch_routes() -> tuple[Route, ...]:
    """Every recognition route `route_for` can return, once each.

    The installer downloads exactly these, so a machine can read every
    language the settings offer without a network connection afterwards.
    """
    seen: dict[tuple[str, str, str], Route] = {}
    codes = _V6_LANGS | _V5_OWN | set(_V4_OWN) | _CYRILLIC | _DEVANAGARI | _ARABIC
    for code in sorted(codes):
        route = route_for(code)
        seen[(route.version, route.model_type, route.lang_type)] = route
    return tuple(seen.values())


def route_model_key(route: Route) -> str:
    """The RapidOCR model key a route resolves to, e.g. `cyrillic_PP-OCRv5_rec_mobile`."""
    from rapidocr.utils.model_resolver import route_to_model_key
    from rapidocr.utils.typings import ModelType, OCRVersion, TaskType

    return route_to_model_key(
        TaskType.REC,
        OCRVersion(route.version),
        route.lang_type,
        ModelType(route.model_type),
    )


def downloadable_models() -> tuple[tuple[Route, str], ...]:
    """(route, model key) for every prefetch route that is not bundled.

    The PP-OCRv6 multilingual model ships in the wheel, so a `V6` route costs
    nothing; every other route is a download on first use.
    """
    out = []
    for route in prefetch_routes():
        if (route.version, route.model_type) == V6:
            continue
        out.append((route, route_model_key(route)))
    return tuple(out)


def downloadable_tiers() -> tuple[tuple[str, str, str], ...]:
    """(task, tier, model key) for the PP-OCRv6 sizes the wheel does not carry.

    The default size is bundled, so `--download-models` only has to fetch the
    other two -- detector and recogniser alike, because a folder set to `tiny`
    or `medium` reads with both.
    """
    out = []
    for tier in MODEL_TIERS:
        if tier == DEFAULT_MODEL_TIER:
            continue
        for task in ("det", "rec"):
            out.append((task, tier, f"{DET_MODEL[0]}_{task}_{tier}"))
    return tuple(out)


# --- the device ------------------------------------------------------------

_DML: bool | None = None
_DML_LOCK = threading.Lock()

_UNSET = object()          # "not passed", so None can mean "no cache dir"

DML = "dml"
CPU = "cpu"
DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def dml_available() -> bool:
    """True when this machine can run ONNX Runtime on DirectML.

    Needs the `DmlExecutionProvider` in the installed onnxruntime, which means
    the `onnxruntime-directml` wheel (the plain `onnxruntime` wheel has only
    the CPU provider) and Windows 10 build 18362 or newer.
    """
    global _DML
    with _DML_LOCK:
        if _DML is None:
            try:
                import onnxruntime

                providers = onnxruntime.get_available_providers()
                _DML = DML_PROVIDER in providers
            except Exception as exc:            # no onnxruntime, or a broken one
                logger.warning("DirectML check failed, OCR will run on the CPU: %s", exc)
                _DML = False
            if not _DML:
                logger.info("no DirectML provider; OCR runs on the CPU")
            else:
                logger.debug("DirectML is available: %s", providers)
        return _DML


def resolve_device(use_gpu) -> str:
    """The device an engine is built on: DirectML when the folder asks for the
    GPU *and* this machine has DirectML, else the CPU. A missing or broken
    DirectML provider no longer fails every engine build just because
    `use_gpu` defaults to True."""
    if not use_gpu:
        return CPU
    return DML if dml_available() else CPU


def provider_names(engine) -> list[str]:
    """The execution providers an engine's three sessions actually got, first
    one first. Empty entries are dropped: a sub-engine that was never used has
    not built its session yet."""
    names: list[str] = []
    for attribute in ("text_det", "text_cls", "text_rec"):
        session = getattr(engine, attribute, None)
        session = getattr(session, "session", None)
        getter = getattr(session, "session", None)
        providers = list(getattr(getter, "get_providers", lambda: [])()) if getter else []
        for name in providers:
            if name not in names:
                names.append(name)
    return names


# --- the engine ------------------------------------------------------------


def default_model_dir():
    """Where RapidOCR caches the models that are not in the wheel.

    `<data>/models` (`core.runtime.paths.models_dir()`), not rapidocr's own
    site-packages: a bundle's install directory must not be written to, and
    a per-user cache survives an app update. `None` when the data dir cannot
    be resolved, which leaves RapidOCR's default in place.
    """
    try:
        from core.runtime.paths import models_dir

        return models_dir()
    except Exception as exc:                        # noqa: BLE001 - a cache path is not worth failing over
        logger.warning("could not resolve the model cache directory: %s", exc)
        return None


def model_dirs() -> tuple[Path, ...]:
    """Every directory a model may already sit in, most specific first.

    1. `<checkout>/models` -- a developer checkout that fetched the extra
       sizes once (the same shape the bundle uses);
    2. `<checkout>/../models` -- the bundle's own `models/`, when the sources
       live in `<bundle>/src` beside `bin/`;
    3. the per-user cache (`default_model_dir()`), where RapidOCR downloads
       anything it does not find above.

    The wheel's own `rapidocr/models/` is handled separately in
    `bundled_model`, because it is looked up through the package rather than
    through a path.
    """
    dirs: list[Path] = []
    try:
        from core.runtime.paths import source_root

        root = Path(source_root())
        dirs.extend((root / "models", root.parent / "models"))
    except Exception:                               # noqa: BLE001 - no layout, no bundle models
        pass
    cache = default_model_dir()
    if cache is not None:
        dirs.append(Path(cache))
    return tuple(dirs)


def bundled_model(name: str):
    """The absolute path of a model this build already has on disk.

    Searched in this order: inside the rapidocr wheel (`rapidocr/models/`,
    which carries the PP-OCRv6 `small` detector, recogniser and the
    orientation classifier), then the app's own `models/` folders -- where the
    released Windows build ships all three PP-OCRv6 sizes -- then the per-user
    cache that RapidOCR downloads into. See `model_dirs`.

    None when no copy exists, which leaves RapidOCR to fetch that model into
    the cache itself.
    """
    candidates = []
    try:
        import rapidocr

        candidates.append(Path(rapidocr.__file__).resolve().parent / "models" / name)
    except Exception:                               # noqa: BLE001 - a missing wheel model is not fatal
        pass
    candidates.extend(directory / name for directory in model_dirs())
    for path in candidates:
        if path.is_file():
            return path
    return None


def _params(device: str, route: Route | None, model_root_dir,
            tier: str = DEFAULT_MODEL_TIER) -> dict:
    """RapidOCR's `params` dict. Values for the enum-typed keys MUST be the
    enum members, not strings: `ParseParams.update_batch` rejects a string with
    "The value of Det.model_type must be Enum Type.".

    `tier` picks the PP-OCRv6 size for both the detector and (when the
    language is covered by PP-OCRv6) the recogniser. A route pinned to another
    version -- PP-OCRv5 for Cyrillic, say -- keeps its own model and is
    unaffected: those versions have no size tiers here.
    """
    from rapidocr import ModelType, OCRVersion

    tier = normalise_tier(tier)
    det_version, det_type = DET_MODEL
    det_type = tier
    # The size applies to PP-OCRv6 only. A route on another version -- PP-OCRv5
    # for Cyrillic, PP-OCRv4 for a few scripts -- has no tiny/medium weights,
    # so it keeps its own and the picker does not change its accuracy.
    rec_type = tier if (route is not None and (route.version, route.model_type) == V6) \
        else (route.model_type if route is not None else None)
    params = {
        "Global.log_level": "warning",
        "EngineConfig.onnxruntime.use_dml": device == DML,
        "Det.ocr_version": OCRVersion(det_version),
        "Det.model_type": ModelType(det_type),
        "Cls.model_type": ModelType(BUNDLED_CLS[1]),
    }
    if route is not None:
        params["Rec.ocr_version"] = OCRVersion(route.version)
        params["Rec.model_type"] = ModelType(rec_type)
        params["Rec.lang_type"] = route.lang_type

    # The bundled models are named explicitly rather than left to
    # `Global.model_root_dir`, which takes precedence over the wheel: setting
    # only the cache directory re-downloads the 30 MB PP-OCRv6 pair the wheel
    # already carries (measured: det 9.47 MB + cls 0.56 MB + rec 20.25 MB on
    # the first engine build). A model that is NOT there -- a size the wheel
    # does not carry, or a language on another PP-OCR version -- is left to
    # RapidOCR, which downloads it into `Global.model_root_dir` on first use.
    # The released Windows bundle ships all three sizes in that same folder,
    # so a packaged build never downloads at all.
    for section, name in (("Det", _bundled_name("det", det_version, det_type)),
                          ("Cls", BUNDLED_CLS_FILE)):
        path = bundled_model(name)
        if path is not None:
            params[f"{section}.model_path"] = str(path)
    if route is not None:
        path = bundled_model(_bundled_name("rec", route.version, tier))
        if path is not None:
            params["Rec.model_path"] = str(path)
    if model_root_dir:
        params["Global.model_root_dir"] = str(model_root_dir)
    logger.debug("OCR params: device=%s tier=%s route=%s -> det %s/%s, rec %s/%s, "
                 "bundled det=%s rec=%s", device, tier,
                 None if route is None else f"{route.lang_type}@{route.version}",
                 det_version, det_type,
                 None if route is None else route.version, None if route is None else rec_type,
                 params.get("Det.model_path"), params.get("Rec.model_path"))
    return params


def _bundled_name(task: str, version: str, model_type: str) -> str:
    """The file name rapidocr's wheel uses for one of its models.

    `PP-OCRv6` + `det` + `small` -> "PP-OCRv6_det_small.onnx".
    """
    return f"{version}_{task}_{model_type}.onnx"


class RapidEngine:
    """A RapidOCR engine with the PaddleOCR 3.x calling convention.

    `predict(images)` takes a *list* of BGR frames -- RapidOCR itself reads one
    image per call -- and yields one result dict per image, so
    `list(engine.predict(frames))` returns the same list of dicts the rest of
    the pipeline has always consumed. `ocr(images)` is the same thing under the
    name the pre-3.0 code used.
    """

    def __init__(self, engine, device: str, route: Route | None = None, kind: str = "ocr",
                 tier: str = DEFAULT_MODEL_TIER):
        self.engine = engine
        self.device = device
        self.route = route
        self.kind = kind
        self.tier = normalise_tier(tier)
        self.providers = provider_names(engine)

    def __repr__(self) -> str:                       # pragma: no cover - debugging aid
        route = f" route={self.route.lang_type}@{self.route.version}" if self.route else ""
        return f"<RapidEngine {self.kind} {self.device} {self.tier}{route} {self.providers}>"

    def warm_up(self) -> list[str]:
        """Build every session this engine needs and return their providers.

        RapidOCR builds each sub-engine on first use, so a freshly constructed
        engine reports no providers yet. The installer's verify step calls this
        to prove DirectML really is in play before the engine is accepted --
        and, for a language whose recognition model is not bundled, it is also
        what downloads that model.
        """
        loaders = ["_load_det_model"]
        if self.kind != "det":
            loaders += ["_load_cls_model", "_load_rec_model"]
        for name in loaders:
            loader = getattr(self.engine, name, None)
            if loader is None:                  # pragma: no cover - RapidOCR internals moved
                logger.debug("RapidOCR has no %s; skipping it", name)
                continue
            try:
                loader()
            except Exception as exc:            # a session that cannot load is the failure we want
                raise RuntimeError(f"{name} failed: {exc}") from exc
        self.providers = provider_names(self.engine)
        return self.providers

    # -- full read ---------------------------------------------------------

    def _read(self, image: np.ndarray) -> dict:
        result = self.engine(image)
        texts = getattr(result, "txts", None)
        if texts is None:                  # nothing found: a bare RapidOCROutput
            logger.debug("OCR read %dx%d: no text (detection found nothing above the "
                         "score threshold)", image.shape[1], image.shape[0])
            return {"rec_texts": [], "rec_scores": [], "rec_polys": []}
        boxes = getattr(result, "boxes", None)
        scores = getattr(result, "scores", None)
        out = {
            "rec_texts": [str(text) for text in texts],
            "rec_scores": [float(score) for score in scores],
            "rec_polys": [
                np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes
            ],
        }
        logger.debug("OCR read %dx%d: %d line(s), scores %s, texts %r",
                     image.shape[1], image.shape[0], len(out["rec_texts"]),
                     [round(s, 4) for s in out["rec_scores"]], out["rec_texts"])
        return out

    # -- detection only ----------------------------------------------------

    def _detect(self, image: np.ndarray) -> dict:
        result = self.engine(image)
        boxes = getattr(result, "boxes", None)
        if boxes is None:                  # a bare RapidOCROutput: no text found
            logger.debug("text detection on %dx%d: nothing found", image.shape[1], image.shape[0])
            return {"dt_polys": np.zeros((0, 4, 2), dtype=np.float32), "dt_scores": []}
        scores = getattr(result, "scores", None)
        out = {
            "dt_polys": np.asarray(boxes, dtype=np.float32).reshape(-1, 4, 2),
            "dt_scores": [float(score) for score in scores],
        }
        logger.debug("text detection on %dx%d: %d box(es), scores %s, first box %s",
                     image.shape[1], image.shape[0], len(out["dt_scores"]),
                     [round(s, 4) for s in out["dt_scores"]],
                     out["dt_polys"][0].tolist() if len(out["dt_polys"]) else None)
        return out

    def predict(self, images) -> Iterator[dict]:
        """One result dict per image, in order.

        `images` is normally a list of BGR frames, but every PaddleOCR call
        site also allowed a single frame (`ocr.predict(image)` and
        `det_engine.predict(frame)` are both used in this codebase), so a bare
        ndarray is treated as one image rather than iterated row by row.
        """
        read = self._detect if self.kind == "det" else self._read
        if isinstance(images, np.ndarray):
            images = [images]
        for image in images:
            yield read(image)

    def ocr(self, images) -> list[dict]:
        return list(self.predict(images))


def _build(params: dict, device: str, route: Route | None, kind: str,
           tier: str = DEFAULT_MODEL_TIER) -> RapidEngine:
    from rapidocr import RapidOCR

    from .utils import suppress_output

    with suppress_output():
        engine = RapidOCR(params=params)
    built = RapidEngine(engine, device, route, kind, tier)

    # RapidOCR only warns when DirectML was asked for and not given, then runs
    # on the CPU. Ask for a provider and check we got it: a silent 20x
    # slowdown is worse than a log line.
    wanted = DML_PROVIDER if device == DML else CPU_PROVIDER
    got = built.providers
    if device == DML and got and wanted not in got:
        logger.warning(
            "DirectML was requested but the session got %s; OCR runs on the CPU", got
        )
    logger.info(
        "OCR engine ready: %s, %s model, providers %s",
        device, built.tier, ", ".join(got) or "none built yet",
    )
    return built


def create_ocr_engine(lang, det_model_dir=None, rec_model_dir=None, use_gpu=True,
                      model_root_dir=_UNSET, tier=DEFAULT_MODEL_TIER):
    """A full engine (detection + recognition) for `lang`.

    `det_model_dir` / `rec_model_dir` are PaddleOCR's custom-model arguments and
    are accepted for call-site compatibility only: RapidOCR resolves its own
    models, and the app has no UI for either.

    `model_root_dir` is where a model that is not in the wheel is cached (and
    downloaded to on first use); it defaults to `default_model_dir()`.

    `tier` is the PP-OCRv6 size -- "tiny", "small" or "medium"
    (`MODEL_TIERS`). The default "small" is the pair the wheel carries, so a
    folder that never changes it downloads nothing; the other two are fetched
    into the cache once. An unknown value falls back to the default.
    """
    device = resolve_device(use_gpu)
    route = route_for(lang)
    if det_model_dir or rec_model_dir:
        logger.info(
            "custom model directories are not used by RapidOCR (%s, %s)",
            det_model_dir, rec_model_dir,
        )
    root = default_model_dir() if model_root_dir is _UNSET else model_root_dir
    tier = normalise_tier(tier)
    return _build(_params(device, route, root, tier), device, route, "ocr", tier)


def create_detection_engine(det_model_dir=None, use_gpu=True, tier=DEFAULT_MODEL_TIER):
    """A detection-only engine for label scanning.

    The PP-OCRv6 detector reads every script, so this needs no language. `tier`
    picks its size, exactly as `create_ocr_engine` does.
    """
    device = resolve_device(use_gpu)
    tier = normalise_tier(tier)
    params = _params(device, None, None, tier)
    params.update({"Global.use_det": True, "Global.use_cls": False, "Global.use_rec": False})
    return _build(params, device, None, "det", tier)


def _cancelled(cancel) -> bool:
    """`cancel` is either a zero-argument predicate or a threading.Event."""
    if cancel is None:
        return False
    checker = getattr(cancel, "is_set", None)
    if callable(checker):
        return bool(checker())
    return bool(cancel())


def download_models(model_root_dir=None, progress=None, cancel=None) -> list[str]:
    """Fetch every model a folder could ask for that is not already on disk
    into `model_root_dir`.

    Two things are fetched: the recognition model of every language whose
    route PP-OCRv6 does not cover, and the PP-OCRv6 detector and recogniser
    for the non-default sizes (`tiny`, `medium`). Returns the model keys that
    had to be fetched. `model_root_dir` defaults to `default_model_dir()`, the
    directory the engines read from, so a download here is a download they
    will find. `cancel` is a predicate or a `threading.Event`. Raises on
    failure, and `RuntimeError("cancelled")` when cancelled.
    """
    model_root_dir = default_model_dir() if model_root_dir is None else model_root_dir
    if model_root_dir is None:
        raise RuntimeError("no model cache directory to download into")
    models = downloadable_models()
    tiers = downloadable_tiers()
    if not models and not tiers:
        return []

    from rapidocr import ModelType, OCRVersion, RapidOCR

    from .utils import suppress_output

    total = len(models) + len(tiers)
    fetched: list[str] = []

    def report(index: int, key: str) -> None:
        if progress is not None:
            progress(index, total, key)

    for index, (route, key) in enumerate(models):
        if _cancelled(cancel):
            raise RuntimeError("cancelled")
        report(index, key)
        params = {
            "Global.log_level": "warning",
            "Global.model_root_dir": str(model_root_dir),
            "EngineConfig.onnxruntime.use_dml": False,
            "Det.ocr_version": OCRVersion(DET_MODEL[0]),
            "Det.model_type": ModelType(DET_MODEL[1]),
            "Rec.ocr_version": OCRVersion(route.version),
            "Rec.model_type": ModelType(route.model_type),
            "Rec.lang_type": route.lang_type,
        }
        with suppress_output():
            # Models load lazily, so ask for the recogniser directly: that is
            # what downloads, and it skips building the detector's session.
            RapidOCR(params=params)._load_rec_model()
        fetched.append(key)

    for offset, (task, tier, key) in enumerate(tiers):
        if _cancelled(cancel):
            raise RuntimeError("cancelled")
        report(len(models) + offset, key)
        params = {
            "Global.log_level": "warning",
            "Global.model_root_dir": str(model_root_dir),
            "EngineConfig.onnxruntime.use_dml": False,
            "Det.ocr_version": OCRVersion(DET_MODEL[0]),
            "Det.model_type": ModelType(tier),
            "Rec.ocr_version": OCRVersion(V6[0]),
            "Rec.model_type": ModelType(tier),
            "Rec.lang_type": LangRec.CH.value if hasattr(LangRec.CH, "value") else "ch",
        }
        with suppress_output():
            engine = RapidOCR(params=params)
            # Ask for the half this entry is about: building both would fetch
            # the other one twice.
            if task == "det":
                engine._load_det_model()
            else:
                engine._load_rec_model()
        fetched.append(key)
    return fetched
