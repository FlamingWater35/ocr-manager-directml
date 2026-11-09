"""Filesystem failures, injected rather than arranged.

Several tests pin the same contract: a cache or a save that cannot write must
degrade to "do it again next time", never raise. The original way to arrange
that was `chmod(0o500)` on the directory, which only works where the OS lets
you deny yourself access. Windows' `chmod` only toggles the read-only
attribute of the directory entry; the owner keeps full access to its
children, so nothing fails and the test asserts a lie.

These helpers inject the failure at the syscall the code actually writes
through instead. That is the same `OSError` a read-only mount or a full disk
produces, raised one layer up, and it works on every platform -- so the
contract stays pinned on the only platform this app ships.

`tests/test_project_model.py` already injects `os.fsync`/`os.replace` the same
way for its atomic-write ordering test; this module is that idea, shared.
"""
from __future__ import annotations

from pathlib import Path


def _under(path: str | Path, root: str | Path) -> bool:
    """Whether `path` is inside `root` (case-insensitively on Windows, where
    the same directory has many spellings)."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True


def deny_writes(directory: str | Path, monkeypatch) -> None:
    """Make every atomic write inside `directory` fail with PermissionError.

    Wraps `os.replace`, which is the last step of both writers that matter
    (`core.project.store._atomic_write_text` and
    `core.jobs.view_cache._atomic_write`): the temporary file is written and
    fsynced, then the rename onto the final name fails, exactly as it does on
    a read-only mount or a full disk. Every other path is left alone, so a
    test can still assert what the *rest* of the save did.
    """
    import os

    real_replace = os.replace

    def replace(src, dst, *args, **kwargs):
        if _under(dst, directory):
            raise PermissionError(13, "Permission denied (injected)", str(dst))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)


def deny_removals(directory: str | Path, monkeypatch) -> None:
    """Make deleting anything inside `directory` fail with PermissionError --
    the stale-evidence and view-cache sweeps' version of a read-only mount."""
    import os

    real_remove = os.remove

    def remove(path, *args, **kwargs):
        if _under(path, directory):
            raise PermissionError(13, "Permission denied (injected)", str(path))
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", remove)


def deny_rmtree(monkeypatch) -> None:
    """Make `shutil.rmtree` fail with PermissionError, for the one caller that
    removes a whole directory (`core.jobs.view_cache.prune`)."""
    import shutil

    def rmtree(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied (injected)", str(path))

    monkeypatch.setattr(shutil, "rmtree", rmtree)
