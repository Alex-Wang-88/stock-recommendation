"""Cross-process locks for local synchronization jobs.

The web app's ``threading.Lock`` only protects one Streamlit process.  The
project can also be started by Task Scheduler or from a second terminal, so
the actual synchronization boundary needs a filesystem lock as well.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class SyncAlreadyRunning(RuntimeError):
    """Another process currently owns the project's synchronization lock."""


@contextmanager
def sync_lock(path: str | Path) -> Iterator[None]:
    """Acquire an exclusive, non-blocking lock file for one synchronization run.

    ``msvcrt`` is used on Windows, while ``fcntl`` keeps the same behavior when
    the test suite is run on a POSIX host.  The lock file is deliberately kept
    in ``logs/``; it is runtime state, not a data artifact.
    """

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise SyncAlreadyRunning(f"同步正在运行：{lock_path}") from error
        else:  # pragma: no cover - exercised only on POSIX hosts
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise SyncAlreadyRunning(f"同步正在运行：{lock_path}") from error
        acquired = True
        yield
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised only on POSIX hosts
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        else:
            handle.close()


__all__ = ["SyncAlreadyRunning", "sync_lock"]
