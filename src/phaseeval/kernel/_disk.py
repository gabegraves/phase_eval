from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from phaseeval.kernel.errors import Conflict, TooLarge


def directory(path: Path, *, create: bool = False) -> None:
    for current in (*reversed(path.parents), path):
        if create:
            try:
                current.mkdir(mode=0o700)
                sync_directory(current.parent)
            except FileExistsError:
                pass
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise ValueError("custody directory is unsafe")


def read_capped(path: Path, max_bytes: int) -> bytes:
    directory(path.parent)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as handle:
        metadata = os.fstat(handle.fileno())
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("custody file is not a regular file")
        if metadata.st_size > max_bytes:
            raise TooLarge()
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise TooLarge()
    return data


def sync_directory(path: Path) -> None:
    if os.name == 'nt':
        return  # Windows does not support opening a directory for fsync.
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_once(target: Path, data: bytes) -> bool:
    directory(target.parent, create=True)
    fd, name = tempfile.mkstemp(prefix='.', suffix='.partial', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            try:
                existing = read_capped(target, len(data))
            except TooLarge:
                raise Conflict() from None
            if existing != data:
                raise Conflict()
            return False
        sync_directory(target.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def children(path: Path) -> tuple[Path, ...]:
    try:
        directory(path)
    except FileNotFoundError:
        return ()
    return tuple(sorted(p for p in path.iterdir() if not p.name.startswith('.')))
