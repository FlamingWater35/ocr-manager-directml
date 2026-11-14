"""The release artifacts are named once and agreed on in four places.

`packaging/build.py` writes them, the release workflow globs them twice (the
upload-artifact step and `sha256sum`), lists them in the release notes, and the
README tells the reader what to look for. Renaming the artifacts in only one of
those is silent: the build succeeds, the upload step fails with "no files
found", or -- worse -- the notes advertise a file the release does not carry.

So the name is spelled once, in `build.artifact_stem`, and the other three are
read back and compared against it. These tests never run ISCC or a real build.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _load_build():
    """`packaging/` is not a package (no `__init__.py`), so load it by path."""
    spec = importlib.util.spec_from_file_location("pkg_build_for_names", ROOT / "packaging" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build()


def test_the_stem_carries_the_edition_the_version_and_the_platform():
    """`ocr-manager-directml-v3.1.0-windows`.

    The edition has to be in the file name because the upstream `ocr-manager`
    ships its own `ocr-manager-v*-win` downloads: on a releases page the two
    would differ only by which repository you are looking at."""
    assert build.artifact_stem("3.1.0") == "ocr-manager-directml-v3.1.0-windows"
    assert build.artifact_stem("10.0.12") == "ocr-manager-directml-v10.0.12-windows"


def test_the_build_writes_both_artifacts_from_that_stem():
    """The .zip and the .exe share the stem, and the zip's entries sit under a
    folder of the same name, so an unzip does not scatter 500 MB into cwd."""
    source = (ROOT / "packaging" / "build.py").read_text(encoding="utf-8")
    assert "stem = artifact_stem(version)" in source
    assert 'archive = dist / f"{stem}.zip"' in source
    assert 'f"/DOutputBaseFilename={stem}"' in source
    assert 'zf.write(path, f"{stem}/{path.relative_to(tree).as_posix()}")' in source


def test_the_workflow_globs_the_same_prefix():
    """A stale glob is the failure this pins: `upload-artifact` has
    `if-no-files-found: error`, so the build job dies after a full 20-minute
    bundle with no artifact to show for it."""
    prefix = build.ARTIFACT_PREFIX + "-v*"
    assert f"path: dist/{prefix}" in WORKFLOW
    assert f"sha256sum {prefix} > SHA256SUMS.txt" in WORKFLOW
    assert f"dist/{prefix}" in WORKFLOW
    assert "dist/ocr-manager-v*" not in WORKFLOW


def test_the_release_notes_name_the_files_the_release_carries():
    """The notes table is built by hand inside a heredoc, so nothing else
    checks it. `$TAG` is `v3.1.0`, and `artifact_stem` already adds the `v`,
    which makes the two lines agree only if both are spelled the same way.
    The heredoc escapes its backticks, so they are unescaped first."""
    tag = "v3.1.0"
    stem = build.artifact_stem(tag[1:])
    notes = WORKFLOW.replace("$TAG", tag).replace("\\`", "`")
    assert f"`{stem}.exe`" in notes
    assert f"`{stem}.zip`" in notes


def test_the_readme_lists_the_same_two_names():
    """The README is the only place a user learns what the download is called."""
    stem = build.artifact_stem("<version>")
    assert f"`{stem}.exe`" in README
    assert f"`{stem}.zip`" in README
    assert not re.search(r"`ocr-manager-(?!directml)[^`]*-win\.", README)
