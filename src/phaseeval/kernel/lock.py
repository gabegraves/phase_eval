from __future__ import annotations

import contextlib
import ctypes
import errno
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

LOCK_FILE_NAME = ".lock"


def _host_id() -> str:
    if sys.platform.startswith("linux"):
        return Path("/etc/machine-id").read_text().strip()
    if sys.platform == "darwin":
        buf = ctypes.create_string_buffer(16)
        timeout = (ctypes.c_long * 2)(1, 0)
        if ctypes.CDLL(None, use_errno=True).gethostuuid(buf, timeout):
            raise OSError("Cannot read macOS host identity")
        return buf.raw.hex()
    return socket.gethostname().casefold()


def _boot_id() -> str:
    if sys.platform.startswith("linux"):
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if sys.platform == "darwin":
        return _start_token(1)  # launchd start timeval identifies this boot, including in a sandbox
    if sys.platform == "win32":
        # SYSTEM_TIMEOFDAY_INFORMATION starts with BootTime (100 ns since 1601).
        buf = ctypes.create_string_buffer(48)
        status = ctypes.WinDLL("ntdll").NtQuerySystemInformation(3, buf, len(buf), None)
        if status != 0:
            raise OSError("Cannot read Windows boot identity")
        return str(ctypes.c_int64.from_buffer(buf).value)
    raise OSError(f"Unsupported ownership platform: {sys.platform}")


def _windows_times(pid: int) -> str:
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 5:
            raise PermissionError(pid)
        return ""
    try:
        code = wintypes.DWORD()
        times = [wintypes.FILETIME() for _ in range(4)]
        if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise PermissionError(pid)
        if code.value != 259:
            return ""
        if not api.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise PermissionError(pid)
        return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
    finally:
        api.CloseHandle(handle)


def _start_token(pid: int) -> str:
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return "" if fields[0] == "Z" else fields[19]
        except (FileNotFoundError, ProcessLookupError):
            return ""
    if sys.platform == "win32":
        return _windows_times(pid)
    if sys.platform == "darwin":
        # kinfo_proc starts with extern_proc.p_un.__p_starttime (struct timeval).
        libc = ctypes.CDLL(None, use_errno=True)
        mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
        buf = ctypes.create_string_buffer(4096)
        size = ctypes.c_size_t(len(buf))
        if libc.sysctl(mib, 4, buf, ctypes.byref(size), None, 0):
            error = ctypes.get_errno()
            if error == errno.ESRCH:
                return ""
            raise OSError(error, "KERN_PROC_PID")
        if not size.value or buf.raw[36] == 5:  # extern_proc.p_stat == SZOMB on 64-bit Darwin
            return ""
        return bytes(buf[:2 * ctypes.sizeof(ctypes.c_long)]).hex()
    raise OSError(f"Unsupported ownership platform: {sys.platform}")


class ProcessOwner(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    pid: int = Field(ge=1)
    start_token: str = Field(min_length=1, max_length=64)
    host_id: str = Field(default_factory=_host_id)
    boot_id: str = Field(default_factory=_boot_id)

    @classmethod
    def current(cls) -> ProcessOwner:
        return cls.for_pid(os.getpid())

    @classmethod
    def for_pid(cls, pid: int) -> ProcessOwner:
        return cls(pid=pid, start_token=_start_token(pid))

    def alive(self) -> bool:
        if self.host_id != _host_id():
            raise RuntimeError("Store belongs to another host; only single-host local stores are supported")
        if self.boot_id != _boot_id():
            return False
        try:
            return self.start_token == _start_token(self.pid)
        except PermissionError:
            return True  # fail closed: unreadable identity must never permit takeover


def _require_local(root: Path) -> None:
    if sys.platform == "darwin":
        # Darwin statvfs exposes POSIX flags only; statfs64 retains MNT_LOCAL.
        buf = ctypes.create_string_buffer(4096)
        if ctypes.CDLL(None, use_errno=True).statfs64(os.fsencode(root), buf):
            raise OSError(ctypes.get_errno(), "statfs64")
        if not ctypes.c_uint32.from_buffer(buf, 64).value & 0x1000:  # MNT_LOCAL
            raise RuntimeError("Store requires a local filesystem; network stores are unsupported")
    elif sys.platform.startswith("linux"):
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount = Path(left.split()[4].replace("\\040", " ").replace("\\134", "\\"))
            if root.is_relative_to(mount):
                mounts.append((len(str(mount)), right.split()[0]))
        # ponytail: support only verified local filesystem types; extend after lock/rename checks.
        local = {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "overlay", "zfs", "f2fs", "ramfs"}
        if not mounts or max(mounts)[1] not in local:
            raise RuntimeError("Store requires a supported local filesystem")
    elif sys.platform == "win32":
        if ctypes.WinDLL("kernel32").GetDriveTypeW(ctypes.c_wchar_p(root.anchor)) not in (3, 6):
            raise RuntimeError("Store requires a local filesystem; network stores are unsupported")
    else:
        raise OSError(f"Unsupported store platform: {sys.platform}")


class StoreLock:
    _registry: dict[tuple[int, Path], StoreLock] = {}
    _registry_lock = threading.Lock()

    def __new__(cls, root: Path):
        key = (os.getpid(), Path(root).resolve())
        with cls._registry_lock:
            if key not in cls._registry:
                obj = super().__new__(cls)
                obj._path = key[1] / LOCK_FILE_NAME
                obj._thread_lock = threading.Lock()
                obj._holder = threading.local()
                cls._registry[key] = obj
            return cls._registry[key]

    @property
    def path(self) -> Path:
        return self._path

    def assert_held(self) -> None:
        assert getattr(self._holder, "held", False), "*_locked helper requires the store lock"

    @contextlib.contextmanager
    def _acquire(self, blocking: bool) -> Iterator[bool]:
        if getattr(self._holder, "held", False):
            raise RuntimeError("StoreLock is not reentrant: a transaction is already open on this thread")
        if not self._thread_lock.acquire(blocking=blocking):
            yield False
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            _require_local(self._path.parent)
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if not _lock_fd(fd, blocking=blocking):
                    yield False
                    return
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    raw = os.read(fd, 4096).strip(b"\0")
                    if raw and json.loads(raw)["host_id"] != _host_id():
                        raise RuntimeError("Store lock names another host; only single-host local stores are supported")
                    if not raw:
                        os.lseek(fd, 0, os.SEEK_SET)
                        os.write(fd, json.dumps({"host_id": _host_id(), "boot_id": _boot_id()}).encode())
                        os.fsync(fd)
                    self._holder.held = True
                    try:
                        yield True
                    finally:
                        self._holder.held = False
                finally:
                    _unlock_fd(fd)
            finally:
                os.close(fd)
        finally:
            self._thread_lock.release()

    @contextlib.contextmanager
    def held(self) -> Iterator[None]:
        with self._acquire(True):
            yield

    def try_held(self) -> contextlib.AbstractContextManager[bool]:
        return self._acquire(False)


def _lock_fd(fd: int, *, blocking: bool) -> bool:
    if sys.platform == "win32":
        import msvcrt
        # Windows can lock beyond EOF, so initialization stays inside the lock.
        while True:
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if not blocking:
                    return False
                time.sleep(0.01)
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:
        return False
    return True


def _unlock_fd(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
