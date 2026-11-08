"""Where ffmpeg and ffprobe come from (core/ffmpeg.py).

The app cannot decode video without them, so the rules that decide *which*
copy is used matter: the bundled one must win over whatever is installed
system-wide, an explicit override must win over both, and a machine with
nothing anywhere must produce a message a user can act on rather than a
crash inside a subprocess.
"""
import os
import sys
from pathlib import Path

import pytest

from core import ffmpeg as ffmpeg_tool

EXE = ".exe" if sys.platform == "win32" else ""


def _fake_tools(folder: Path) -> Path:
    """A folder holding an executable-looking ffmpeg and ffprobe."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in ffmpeg_tool.TOOL_NAMES:
        path = folder / f"{name}{EXE}"
        path.write_bytes(b"\x00")
        path.chmod(0o755)
    return folder


@pytest.fixture
def no_env_override(monkeypatch):
    monkeypatch.delenv(ffmpeg_tool.FFMPEG_DIR_ENV, raising=False)


def test_the_checkout_bin_folder_is_used_without_any_environment(no_env_override):
    """Running from source finds <checkout>/bin, which is where the app's own
    copy lives; the launcher is not involved when a developer runs main.py."""
    found = ffmpeg_tool.ffmpeg_path()
    assert found is not None, ffmpeg_tool.missing_message()
    assert Path(found).is_file()
    assert Path(found).parent == Path(__file__).resolve().parent.parent / "bin"
    assert ffmpeg_tool.find_tool("ffmpeg")[1] == ffmpeg_tool.ORIGIN_CHECKOUT


def test_the_override_names_the_folder_to_use(tmp_path, monkeypatch):
    folder = _fake_tools(tmp_path / "elsewhere")
    monkeypatch.setenv(ffmpeg_tool.FFMPEG_DIR_ENV, str(folder))
    assert ffmpeg_tool.ffmpeg_path() == str(folder / f"ffmpeg{EXE}")
    assert ffmpeg_tool.ffprobe_path() == str(folder / f"ffprobe{EXE}")
    assert ffmpeg_tool.find_tool("ffmpeg")[1] == ffmpeg_tool.ORIGIN_ENV


def test_the_bundle_bin_folder_wins_over_path(tmp_path, monkeypatch, no_env_override):
    """A bundle is self-contained: its own ffmpeg decodes, even when the
    machine has a different one on PATH."""
    root = tmp_path / "bundle"
    _fake_tools(root / "bin")
    (root / "bundle.json").write_text('{"version": "1.0.0", "os": "win", "arch": "x86_64"}')
    monkeypatch.setenv("OCR_MANAGER_BUNDLE", str(root))

    path, origin = ffmpeg_tool.find_tool("ffmpeg")
    assert origin == ffmpeg_tool.ORIGIN_BUNDLE
    assert Path(path) == root / "bin" / f"ffmpeg{EXE}"


def test_a_bundle_whose_metadata_is_missing_is_not_a_bundle(tmp_path, monkeypatch, no_env_override):
    root = tmp_path / "not-a-bundle"
    _fake_tools(root / "bin")
    monkeypatch.setenv("OCR_MANAGER_BUNDLE", str(root))

    _path, origin = ffmpeg_tool.find_tool("ffmpeg")
    assert origin != ffmpeg_tool.ORIGIN_BUNDLE


def test_path_is_the_last_resort(tmp_path, monkeypatch, no_env_override):
    """A developer machine with ffmpeg installed system-wide still works,
    but only after every bundled location has been ruled out."""
    folder = _fake_tools(tmp_path / "system")
    monkeypatch.setenv("PATH", str(folder))
    monkeypatch.delenv("OCR_MANAGER_BUNDLE", raising=False)
    monkeypatch.setattr(ffmpeg_tool, "_candidate_dirs", lambda: [])

    path, origin = ffmpeg_tool.find_tool("ffmpeg")
    assert origin == ffmpeg_tool.ORIGIN_PATH
    assert Path(path) == folder / f"ffmpeg{EXE}"


def test_nothing_anywhere_is_reported_not_raised(tmp_path, monkeypatch, no_env_override):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("OCR_MANAGER_BUNDLE", raising=False)
    monkeypatch.setattr(ffmpeg_tool, "_candidate_dirs", lambda: [])

    assert ffmpeg_tool.ffmpeg_path() is None
    assert ffmpeg_tool.available() is False
    assert ffmpeg_tool.missing_tools() == ["ffmpeg", "ffprobe"]
    message = ffmpeg_tool.missing_message()
    assert "ffmpeg, ffprobe" in message
    assert "bin folder" in message
    assert ffmpeg_tool.FFMPEG_DIR_ENV in message


def test_the_argv_helpers_prefix_the_resolved_path(tmp_path, monkeypatch):
    folder = _fake_tools(tmp_path / "tools")
    monkeypatch.setenv(ffmpeg_tool.FFMPEG_DIR_ENV, str(folder))

    assert ffmpeg_tool.ffmpeg_argv("-version") == [str(folder / f"ffmpeg{EXE}"), "-version"]
    assert ffmpeg_tool.ffprobe_argv("-v", "error") == [str(folder / f"ffprobe{EXE}"), "-v", "error"]


def test_the_argv_helpers_fall_back_to_the_bare_name(tmp_path, monkeypatch, no_env_override):
    """A child process that cannot be started at all reports the miss itself;
    the helpers must not raise where the caller has nothing to fall back to."""
    monkeypatch.setattr(ffmpeg_tool, "_candidate_dirs", lambda: [])
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert ffmpeg_tool.ffmpeg_argv("-version")[0] == "ffmpeg"
    assert ffmpeg_tool.ffprobe_argv("-version")[0] == "ffprobe"


def test_ffmpeg_dir_is_only_returned_when_both_tools_share_a_folder(tmp_path, monkeypatch):
    folder = _fake_tools(tmp_path / "shared")
    monkeypatch.setenv(ffmpeg_tool.FFMPEG_DIR_ENV, str(folder))
    assert ffmpeg_tool.ffmpeg_dir() == str(folder)

    other = tmp_path / "other"
    other.mkdir()
    (other / f"ffprobe{EXE}").write_bytes(b"\x00")
    monkeypatch.setattr(ffmpeg_tool, "ffprobe_path", lambda: str(other / f"ffprobe{EXE}"))
    assert ffmpeg_tool.ffmpeg_dir() is None


def test_describe_names_a_path_and_an_origin_for_each_tool(tmp_path, monkeypatch):
    folder = _fake_tools(tmp_path / "described")
    monkeypatch.setenv(ffmpeg_tool.FFMPEG_DIR_ENV, str(folder))

    report = ffmpeg_tool.describe()
    assert set(report) == set(ffmpeg_tool.TOOL_NAMES)
    for name, entry in report.items():
        assert entry["ok"] is True
        assert entry["origin"] == ffmpeg_tool.ORIGIN_ENV
        assert Path(entry["path"]).name == f"{name}{EXE}"


def test_where_we_looked_can_be_shown_to_a_user(tmp_path, monkeypatch):
    monkeypatch.setenv(ffmpeg_tool.FFMPEG_DIR_ENV, str(tmp_path / "looked"))
    looked = ffmpeg_tool.where_we_looked()
    assert str(tmp_path / "looked") in looked
    assert looked[-1] == "PATH"


def test_the_app_ships_ffmpeg_so_the_bundle_never_needs_a_system_one(tmp_path, monkeypatch, no_env_override):
    """The regression this module exists for: a machine with no ffmpeg on PATH
    must still decode, because the copy in the app's bin folder is found."""
    root = tmp_path / "bundle"
    _fake_tools(root / "bin")
    (root / "bundle.json").write_text('{"version": "1.0.0", "os": "win", "arch": "x86_64"}')
    monkeypatch.setenv("OCR_MANAGER_BUNDLE", str(root))
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    assert ffmpeg_tool.available() is True


def test_the_packager_puts_both_binaries_in_the_bundle():
    """packaging/build.py copies ffmpeg and ffprobe into <root>/bin; the
    resolver above looks there, so the two have to agree on the folder."""
    source = (Path(__file__).resolve().parent.parent / "packaging" / "build.py").read_text(encoding="utf-8")
    assert 'bin_dir = root / "bin"' in source
    assert '"ffmpeg.exe"' in source and '"ffprobe.exe"' in source


def test_the_launcher_puts_the_bin_folder_on_path():
    """The launcher and the resolver must agree: the launcher prepends
    <root>\\bin, and core.ffmpeg looks in <root>/bin first."""
    source = (Path(__file__).resolve().parent.parent / "packaging" / "windows" / "launcher.c").read_text(
        encoding="utf-8")
    assert "bin;" in source
