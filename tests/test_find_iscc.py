"""The release build must find Inno Setup where it is actually installed.

`windows_artifacts()` used to hard-code `C:\\Program Files (x86)\\Inno Setup 6`
as the fallback when `iscc` is not on PATH. Inno Setup 7 installs to
`C:\\Program Files\\Inno Setup 7`, so on a machine with only version 7 the
build died after writing the zip, with a bare "no such file" from
`subprocess`. These tests pin the search order and the error a missing
compiler produces.

They do not run ISCC or need it installed: the search is a pure function of
`shutil.which` and the existence of a few paths, both of which are stubbed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_build():
    """`packaging/` is not a package (no __init__.py), so load it by path."""
    spec = importlib.util.spec_from_file_location("pkg_build_for_tests", ROOT / "packaging" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build()


def test_iscc_on_path_wins(monkeypatch):
    """A PATH entry is what the developer chose, so it beats the defaults."""
    monkeypatch.setattr(build.shutil, "which", lambda name: r"D:\tools\ISCC.exe" if name == "iscc" else None)
    assert build.find_iscc() == r"D:\tools\ISCC.exe"


def test_inno_setup_7_in_program_files_is_found(monkeypatch):
    """The bug: only version 6's x86 path was tried, and 7 lives elsewhere."""
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    seven = r"C:\Program Files\Inno Setup 7\ISCC.exe"
    monkeypatch.setattr(build.Path, "exists", lambda self: str(self) == seven)
    assert build.find_iscc() == seven


def test_every_known_location_is_tried(monkeypatch):
    """Order matters only in that all four are considered."""
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    tried: list[str] = []

    def exists(self):
        tried.append(str(self))
        return False

    monkeypatch.setattr(build.Path, "exists", exists)
    with pytest.raises(build.CommandFailed):
        build.find_iscc()
    assert tried == list(build.ISCC_LOCATIONS)


def test_a_missing_iscc_says_what_to_install(monkeypatch):
    """The old failure was `FileNotFoundError: [WinError 2]`, which says
    nothing about Inno Setup. The message must name it and the folders."""
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build.Path, "exists", lambda self: False)
    with pytest.raises(build.CommandFailed) as caught:
        build.find_iscc()
    message = str(caught.value)
    assert "Inno Setup" in message
    assert "ISCC.exe" in message
    for location in build.ISCC_LOCATIONS:
        assert location in message


def test_the_installer_script_is_still_the_one_that_ships():
    """`find_iscc` exists to run this file; a rename would break the build
    silently on a machine that has ISCC on PATH."""
    assert (ROOT / "packaging" / "windows" / "installer.iss").exists()
    source = (ROOT / "packaging" / "build.py").read_text(encoding="utf-8")
    assert "installer.iss" in source
    assert "ISCC_LOCATIONS" in source


def test_the_artifact_build_asks_the_finder_rather_than_a_fixed_path():
    """The regression this file exists for.

    Testing `find_iscc` alone is not enough: restoring the old hard-coded
    fallback at the call site leaves every test above green, because the
    helper still works -- nothing calls it. So assert on the function that
    builds the artifacts. Checked with `inspect.getsource`, which reads the
    parsed source rather than the module's current behaviour."""
    import inspect

    body = inspect.getsource(build.windows_artifacts)
    assert "find_iscc()" in body, "windows_artifacts must call find_iscc()"
    assert "Inno Setup" not in body, "the install folder must not be spelled at the call site"
    assert 'shutil.which("iscc")' not in body, "the PATH lookup belongs in find_iscc()"
