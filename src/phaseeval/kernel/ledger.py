from __future__ import annotations

import enum
import hashlib
import re
import shutil
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import ContextManager, Literal

from pydantic import TypeAdapter

from phaseeval.kernel import _disk
from phaseeval.kernel.errors import TooLarge
from phaseeval.kernel.ids import HEX64, SAFE_ID
from phaseeval.kernel.lock import ProcessOwner, StoreLock
from phaseeval.kernel.records import Record, canonical_bytes


class Mode(enum.Enum):
    ONCE = 'once'
    LOG = 'log'
    MARK = 'mark'


Segment = Literal['hex64', 'safe_id', 'utc_sha', 'int']
_PATTERNS = {
    'hex64': HEX64,
    'safe_id': SAFE_ID,
    'utc_sha': r'\d{8}T\d{12}Z-[0-9a-f]{64}\.json',
    'int': r'0|[1-9][0-9]*',
}


@dataclass(frozen=True)
class KeyShape:
    segments: tuple[Segment, ...]

    def validate(self, key: tuple[str, ...], *, prefix: bool = False) -> None:
        if len(key) > len(self.segments) or (not prefix and len(key) != len(self.segments)):
            raise ValueError('wrong number of key segments')
        for value, shape in zip(key, self.segments):
            if not isinstance(value, str) or not re.fullmatch(_PATTERNS[shape], value):
                raise ValueError(f'invalid {shape} key segment')


@dataclass(frozen=True)
class Limits:
    max_record_bytes: int = 32 * 1024 * 1024
    max_blob_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        if min(self.max_record_bytes, self.max_blob_bytes) < 1:
            raise ValueError('limits must be positive')


@dataclass(frozen=True)
class Claim:
    directory: str
    mode: Mode | Literal['blobs', 'legacy', 'scratch']
    record: str | None
    owner: str


@dataclass(frozen=True)
class Published:
    path: PurePosixPath
    existed: bool


class Collection[R]:
    def __init__(self, ledger: Ledger, name: str, record: type[R], mode: Mode, key: KeyShape, *, readonly=False):
        self._ledger, self.name, self.record, self.mode, self.key = ledger, name, record, mode, key
        self._adapter = TypeAdapter(record)
        self._readonly = readonly

    def _path(self, key: tuple[str, ...]) -> Path:
        self.key.validate(key)
        if '{' in self.name:
            return self._ledger._root / self.name.format(*key)
        path = self._ledger._root / self.name
        path = path.joinpath(*key)
        return path.with_name(path.name + '.json') if self.mode is Mode.ONCE and self.record is not bytes else path

    def _data(self, record: R) -> bytes:
        self._adapter.validate_python(record)
        return record if isinstance(record, bytes) else canonical_bytes(record)

    def _publish(self, path: Path, data: bytes) -> Published:
        if self._readonly:
            raise TypeError('legacy collections are read-only')
        if len(data) > self._ledger._limits.max_record_bytes:
            raise TooLarge()
        return Published(PurePosixPath(path.relative_to(self._ledger._root)), not _disk.publish_once(path, data))

    def put(self, key: tuple[str, ...], record: R) -> Published:
        if self.mode is not Mode.ONCE:
            raise TypeError('put requires ONCE mode')
        return self._publish(self._path(key), self._data(record))

    def append(self, key: tuple[str, ...], record: R, *, at: datetime | None = None) -> Published:
        if self.mode is not Mode.LOG:
            raise TypeError('append requires LOG mode')
        data = self._data(record)
        if at is None:
            at = getattr(record, 'created_at_utc', None)
        if at is None:
            raise ValueError('log records need their recorded timestamp')
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError('log timestamps need a timezone')
        at = at.astimezone(UTC)
        name = f'{at:%Y%m%dT%H%M%S%fZ}-{hashlib.sha256(data).hexdigest()}.json'
        return self._publish(self._path(key) / name, data)

    def get_bytes(self, key: tuple[str, ...]) -> bytes | None:
        if self.mode is not Mode.ONCE:
            raise TypeError('get_bytes requires ONCE mode')
        try:
            return _disk.read_capped(self._path(key), self._ledger._limits.max_record_bytes)
        except FileNotFoundError:
            return None

    def get(self, key: tuple[str, ...]) -> R | None:
        if self.mode is not Mode.ONCE:
            raise TypeError('get requires ONCE mode')
        data = self.get_bytes(key)
        return None if data is None else data if self.record is bytes else self._adapter.validate_json(data)

    def entries(self, key: tuple[str, ...]) -> tuple[R, ...]:
        if self.mode is not Mode.LOG:
            raise TypeError('entries requires LOG mode')
        found = []
        for path in _disk.children(self._path(key)):
            if not re.fullmatch(_PATTERNS['utc_sha'], path.name):
                raise ValueError('invalid log entry name')
            data = _disk.read_capped(path, self._ledger._limits.max_record_bytes)
            if not path.name.endswith(f'-{hashlib.sha256(data).hexdigest()}.json'):
                raise ValueError('record bytes disagree with their name')
            found.append(data if self.record is bytes else self._adapter.validate_json(data))
        return tuple(found)

    def keys(self, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
        self.key.validate(prefix, prefix=True)
        parts = self.name.split('/') if '{' in self.name else [*self.name.split('/'), *('{'+str(i)+'}' for i in range(len(self.key.segments)))]
        if '{' not in self.name and self.mode is Mode.ONCE and self.record is not bytes:
            parts[-1] += '.json'

        def walk(path, index, key):
            if index == len(parts):
                yield key
                return
            part = parts[index]
            match = re.fullmatch(r'\{(\d+)\}(.*)', part)
            if match:
                position, suffix = int(match[1]), match[2]
                for child in _disk.children(path):
                    if suffix and not child.name.endswith(suffix):
                        continue
                    value = child.name[:-len(suffix)] if suffix else child.name
                    if not re.fullmatch(_PATTERNS[self.key.segments[position]], value):
                        continue
                    if position < len(prefix) and value != prefix[position]:
                        continue
                    yield from walk(child, index+1, (*key, value))
            else:
                next_path = path / part
                try:
                    next_path.lstat()
                except FileNotFoundError:
                    return
                yield from walk(next_path, index+1, key)
        yield from walk(self._ledger._root, 0, ())


class Markers:
    def __init__(self, ledger: Ledger, name: str, key: KeyShape):
        self._ledger, self.name, self.key = ledger, name, key

    def add(self, key: tuple[str, ...]) -> Published:
        self.key.validate(key)
        path = self._ledger._root / self.name
        path = path.joinpath(*key)
        return Published(PurePosixPath(path.relative_to(self._ledger._root)), not _disk.publish_once(path, b''))

    def contains(self, key: tuple[str, ...]) -> bool:
        self.key.validate(key)
        try:
            _disk.read_capped(self._ledger._root.joinpath(self.name, *key), 0)
        except FileNotFoundError:
            return False
        return True

    def members(self, prefix: tuple[str, ...]) -> tuple[str, ...]:
        self.key.validate(prefix, prefix=True)
        if len(prefix) >= len(self.key.segments):
            raise ValueError('members requires an incomplete key')
        found = []
        for path in _disk.children(self._ledger._root.joinpath(self.name, *prefix)):
            if not re.fullmatch(_PATTERNS[self.key.segments[len(prefix)]], path.name):
                continue
            if len(prefix) == len(self.key.segments) - 1:
                _disk.read_capped(path, 0)
            else:
                _disk.directory(path)
            found.append(path.name)
        return tuple(found)


class Blobs:
    def __init__(self, ledger: Ledger):
        self._ledger = ledger

    def put(self, data: bytes) -> str:
        if len(data) > self._ledger._limits.max_blob_bytes:
            raise TooLarge()
        ref = hashlib.sha256(data).hexdigest()
        _disk.publish_once(self._ledger._root / 'blobs' / ref, data)
        return ref

    def get(self, ref: str) -> bytes:
        KeyShape(('hex64',)).validate((ref,))
        data = _disk.read_capped(self._ledger._root / 'blobs' / ref, self._ledger._limits.max_blob_bytes)
        if hashlib.sha256(data).hexdigest() != ref:
            raise ValueError('blob bytes disagree with their digest')
        return data


class ScratchOwner(Record):
    owner: ProcessOwner
    generation: int | None = None
    slot: str | None = None


class Scratch:
    def __init__(self, ledger: Ledger):
        self._ledger = ledger
        # A claimed slot stays intact until its tree supplies the child-liveness guard.
        self.child_alive: Callable[[str], bool] = lambda slot: True

    def _detach_locked(self, path: Path) -> Path:
        self._ledger.assert_locked()
        _disk.directory(path)
        trash = path.with_name('.trash-' + uuid.uuid4().hex)
        path.rename(trash)
        return trash

    @contextmanager
    def workdir(self, slot: str | None = None) -> Iterator[Path]:
        if slot is not None:
            KeyShape(('hex64',)).validate((slot,))
        path = self._ledger._root / 'scratch' / uuid.uuid4().hex
        with self._ledger.transaction():
            _disk.publish_once(path / 'owner.json', canonical_bytes(ScratchOwner(owner=ProcessOwner.current(), slot=slot)))
        try:
            yield path
        finally:
            with self._ledger.transaction():
                trash = self._detach_locked(path) if slot is None or not self.child_alive(slot) else None
            if trash is not None:
                shutil.rmtree(trash)

    def sweep(self) -> int:
        detached = []
        with self._ledger.transaction():
            for path in _disk.children(self._ledger._root / 'scratch'):
                owner = ScratchOwner.model_validate_json(_disk.read_capped(path / 'owner.json', self._ledger._limits.max_record_bytes))
                if not owner.owner.alive() and (owner.slot is None or not self.child_alive(owner.slot)):
                    detached.append(self._detach_locked(path))
        for path in detached:
            shutil.rmtree(path)
        return len(detached)


class Ledger:
    def __init__(self, root: Path, limits: Limits = Limits()) -> None:
        self._root, self._limits = root.resolve(), limits
        self._lock = StoreLock(root)
        self._claims: dict[str, Claim] = {}

    def transaction(self) -> ContextManager[None]:
        return self._lock.held()

    def assert_locked(self) -> None:
        self._lock.assert_held()

    def _claim(self, name, mode, record, owner):
        if not name or name.startswith('/') or any(part in ('', '.', '..') or '\\' in part or '\x00' in part or ':' in part for part in name.split('/')):
            raise ValueError('invalid collection directory')
        if name in self._claims:
            raise ValueError(f'{name} already claimed by {self._claims[name].owner}')
        self._claims[name] = Claim(name, mode, str(record) if record is not None else None, owner)

    def collection[R](self, name: str, record: type[R], mode: Mode, key: KeyShape, *, owner: str) -> Collection[R]:
        self._claim(name, mode, record, owner)
        return Collection(self, name, record, mode, key)

    def markers(self, name: str, key: KeyShape, *, owner: str) -> Markers:
        self._claim(name, Mode.MARK, None, owner)
        return Markers(self, name, key)

    def blobs(self, *, owner: str) -> Blobs:
        self._claim('blobs', 'blobs', bytes, owner)
        return Blobs(self)

    def scratch(self, *, owner: str) -> Scratch:
        self._claim('scratch', 'scratch', ScratchOwner, owner)
        return Scratch(self)

    def legacy[R](self, name: str, record: type[R], *, reader: str, mode: Mode = Mode.ONCE, key: KeyShape = KeyShape(('safe_id',))) -> Collection[R]:
        if name not in self._claims:
            self._claim(name, 'legacy', record, reader)
        return Collection(self, name, record, mode, key, readonly=True)

    def layout(self) -> tuple[Claim, ...]:
        return tuple(self._claims.values())
