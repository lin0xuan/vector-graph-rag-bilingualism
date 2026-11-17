"""Small cross-platform non-blocking file locks for durable jobs."""
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def exclusive_lock(path):
    """Hold a non-blocking exclusive lock, raising ValueError if busy.

    POSIX uses ``flock``. Windows uses the CRT byte-range lock through
    ``msvcrt``; the lock file is kept open for the entire context.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ValueError("Evaluation directory is already in use") from None
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Evaluation directory is already in use") from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
