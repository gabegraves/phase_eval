"""Locate BGMN locally or install a measured archive with explicit consent."""

from __future__ import annotations

import enum
import hashlib
import os
import platform
import stat
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Platform = Literal["linux-x86_64", "darwin-arm64", "darwin-x86_64", "windows-x86_64"]
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


# ponytail: standalone records until kernel.records lands in P11; adopt Record at the P32 move.
class PlatformArchive(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    platform: Platform
    url: str
    sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=MAX_ARCHIVE_BYTES)
    executables: tuple[str, ...] = Field(min_length=1)
    measured_on: date | None


_URL = "https://cedergrouphub.github.io/dara/_static/bgmnwin_{name}.zip"
_UNIX = ("bgmn", "teil", "eflech", "output", "geomet", "verzerr", "makegeq", "gertest", "lamtest", "index", "plot1", "spacegrp")
_WIN = ("BGMN.EXE", "TEIL.EXE", "EFLECH.EXE", "OUTPUT.EXE", "GDUNINST.EXE", "DIALOG.EXE", "MAKEGEQ-5.1.3.EXE", "GEOMET-5.1.3.EXE", "VERZERR.EXE", "GEOMET.EXE", "INDEX.EXE", "MAKEGEQ.EXE", "EXIT1.EXE", "GERTEST.EXE")

# Trust-on-first-use measurements of DARA's copies, downloaded on 2026-10-26.
ARCHIVES: Mapping[Platform, PlatformArchive] = {
    "linux-x86_64": PlatformArchive(platform="linux-x86_64", url=_URL.format(name="Linux"), sha256="d0f94224a9d422a7fdd13e8d29626e3e6df6ec0700aefe614414aabede7c1869", size_bytes=1590922, executables=_UNIX, measured_on=date(2026, 10, 26)),
    "darwin-arm64": PlatformArchive(platform="darwin-arm64", url=_URL.format(name="Darwin_m1"), sha256="21d191de33f9e1dd73e8f7d4d462bc027e719126f59bc72aec8826d81a904e0a", size_bytes=1406817, executables=_UNIX, measured_on=date(2026, 10, 26)),
    "darwin-x86_64": PlatformArchive(platform="darwin-x86_64", url=_URL.format(name="Darwin"), sha256="4d9433e310afc4629b4d03026b36d96b217597a62598935605ceb373061de5c8", size_bytes=1712714, executables=_UNIX, measured_on=date(2026, 10, 26)),
    "windows-x86_64": PlatformArchive(platform="windows-x86_64", url=_URL.format(name="Windows"), sha256="ab939d1bde0ee88d1608119484563d9cc126e95f3be41a92fa2adfd218e9e026", size_bytes=3095446, executables=_WIN, measured_on=date(2026, 10, 26)),
}
assert set(ARCHIVES) == set(get_args(Platform)), "every Platform has one archive row"


class Consent(enum.StrEnum):
    ASK = "ask"
    DOWNLOAD = "download"
    NONE = "none"


class BgmnProgram(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: Path
    version: str | None
    binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BGMNNotInstalledError(RuntimeError):
    def __init__(self, message: str = 'BGMN is not installed. Configure a local install or set bgmn = "download" in launcher/phaseeval.toml.'):
        super().__init__(message)


class BGMNInstaller:
    INSTALL_DIR = Path.home() / ".axa" / "bgmn"
    EXECUTABLES_UNIX = ("bgmn", "teil", "eflech", "output")
    EXECUTABLES_WIN = ("BGMN.EXE", "TEIL.EXE", "EFLECH.EXE", "OUTPUT.EXE")

    def __init__(self, install_dir: Path | None = None):
        configured = None if install_dir is not None else os.environ.get("PHASEEVAL_BGMN_DIR", os.environ.get("AXA_BGMN_DIR"))
        self.install_dir = install_dir if install_dir is not None else (Path(configured) if configured else self.INSTALL_DIR)

    @property
    def archive(self) -> PlatformArchive:
        system, machine = platform.system().lower(), platform.machine().lower()
        machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
        key = f"{system}-{machine}"
        if key not in ARCHIVES:
            raise BGMNNotInstalledError(f"Unsupported BGMN platform: {key}")
        return ARCHIVES[key]

    @property
    def _executables(self) -> tuple[str, ...]:
        return self.EXECUTABLES_WIN if platform.system() == "Windows" else self.EXECUTABLES_UNIX

    @property
    def _bgmn_win_dir(self) -> Path:
        return self.install_dir / "BGMNwin"

    def is_installed(self) -> bool:
        return all((self._bgmn_win_dir / exe).is_file() for exe in self._executables)

    def get_bgmn_path(self) -> Path:
        if not self.is_installed():
            raise BGMNNotInstalledError()
        return self._bgmn_win_dir / self._executables[0]

    def get_bgmn_dir(self) -> Path:
        return self._bgmn_win_dir

    def download_url(self) -> str:
        return self.archive.url

    def validate_installation(self) -> dict[str, bool]:
        return {exe: (self._bgmn_win_dir / exe).is_file() for exe in self._executables}

    def install(self, force: bool = False) -> Path:
        """Explicitly opt in to downloading this platform's pinned BGMN archive."""
        if self.is_installed() and not force:
            return self._bgmn_win_dir
        row = self.archive
        if row.sha256 is None or row.measured_on is None:
            raise BGMNNotInstalledError("BGMN archive has no measured pin")

        import requests

        self.install_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".bgmn-", dir=self.install_dir.parent) as temporary:
            staging = Path(temporary)
            zip_path = staging / "archive.zip"
            digest, size = hashlib.sha256(), 0
            with requests.get(row.url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with zip_path.open("wb") as output:
                    for chunk in response.iter_content(64 * 1024):
                        size += len(chunk)
                        if size > min(row.size_bytes, MAX_ARCHIVE_BYTES):
                            raise RuntimeError("BGMN archive exceeds size cap")
                        digest.update(chunk)
                        output.write(chunk)
            if digest.hexdigest() != row.sha256:
                raise RuntimeError("BGMN archive sha256 mismatch")

            with zipfile.ZipFile(zip_path) as archive:
                members = archive.infolist()
                for member in members:
                    path = PurePosixPath(member.filename)
                    mode = stat.S_IFMT(member.external_attr >> 16)
                    if (path.is_absolute() or ".." in path.parts or "\\" in member.filename or ":" in member.filename
                            or not path.parts
                            or mode not in (0, stat.S_IFREG, stat.S_IFDIR)):
                        raise RuntimeError("Unsafe BGMN archive path")
                members = [member for member in members if PurePosixPath(member.filename).parts[0] == "BGMNwin"]
                if sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
                    raise RuntimeError("BGMN extracted size exceeds cap")
                archive.extractall(staging / "unpacked", members=members)
            unpacked = staging / "unpacked" / "BGMNwin"
            if not all((unpacked / exe).is_file() for exe in row.executables):
                raise RuntimeError("BGMN archive is missing listed executables")
            for path in unpacked.rglob("*"):
                if path.is_file():
                    path.chmod(0o755 if path.relative_to(unpacked).as_posix() in row.executables else 0o644)

            self.install_dir.mkdir(parents=True, exist_ok=True)
            previous = staging / "previous"
            if self._bgmn_win_dir.exists():
                os.replace(self._bgmn_win_dir, previous)
            try:
                os.replace(unpacked, self._bgmn_win_dir)
            except OSError:
                if previous.exists():
                    os.replace(previous, self._bgmn_win_dir)
                raise
        return self._bgmn_win_dir


def locate_bgmn(search: Sequence[Path]) -> BgmnProgram | None:
    for directory in search:
        program = BGMNInstaller(directory)
        if program.is_installed():
            return BgmnProgram(directory=directory, version=None,
                               binary_sha256=hashlib.sha256(program.get_bgmn_path().read_bytes()).hexdigest())
    return None


def install_bgmn(target: Path, consent: Consent, ask: Callable[[PlatformArchive], bool]) -> BgmnProgram | None:
    consent = Consent(consent)
    if consent is Consent.NONE:
        return None
    program = BGMNInstaller(target)
    row = program.archive
    if row.sha256 is None or row.measured_on is None:
        raise BGMNNotInstalledError("BGMN archive has no measured pin")
    if consent is Consent.ASK and (not sys.stdin.isatty() or not ask(row)):
        return None
    program.install()
    return locate_bgmn([target])
