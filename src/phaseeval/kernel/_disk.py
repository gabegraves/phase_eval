from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from phaseeval.kernel.errors import Conflict, TooLarge


def directory(path: Path, *, create: bool = False) -> None:
    for current in (*reversed(path.parents), path):
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not create:
                raise
            try:
                current.mkdir(mode=0o700)
                sync_directory(current.parent)
            except FileExistsError:
                pass
            metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("custody directory is unsafe")


def read_capped(path: Path, max_bytes: int) -> bytes:
    try:
        directory(path.parent)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or getattr(before, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("custody file is not a regular file")
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                     | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0))
        with os.fdopen(fd, 'rb') as handle:
            metadata = os.fstat(handle.fileno())
            if not os.path.samestat(before, metadata) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("custody file changed during open")
            if metadata.st_size > max_bytes:
                raise TooLarge()
            data = handle.read(max_bytes + 1)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError("custody file is unavailable") from exc
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


def children(path: Path, *, include_hidden: bool = False) -> tuple[Path, ...]:
    try:
        directory(path)
    except FileNotFoundError:
        return ()
    return tuple(sorted(p for p in path.iterdir() if include_hidden or not p.name.startswith('.')))
