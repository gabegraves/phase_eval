from __future__ import annotations

import multiprocessing
import os
import threading
from pathlib import Path

import pytest

from phaseeval.kernel.lock import ProcessOwner, StoreLock
from tests.kernel.concurrency import lock_workers

CTX = multiprocessing.get_context("spawn")


def test_the_lock_excludes_another_process_until_released(tmp_path: Path) -> None:
    holding, release, tried = CTX.Event(), CTX.Event(), CTX.Event()
    with CTX.Manager() as manager:
        results = manager.dict()
        a = CTX.Process(target=lock_workers.holder, args=(str(tmp_path), holding, release))
        b = CTX.Process(target=lock_workers.contender, args=(str(tmp_path), holding, tried, results))
        a.start()
        b.start()
        assert tried.wait(20), "the contender never got to try"
        assert results["first_try"] is False
        release.set()
        a.join(20)
        b.join(20)
        assert (a.exitcode, b.exitcode) == (0, 0)
        assert results["eventually"] is True
    assert (tmp_path / ".lock").exists()


def test_the_lock_is_not_reentrant(tmp_path: Path) -> None:
    lock = StoreLock(tmp_path)
    with lock.held():
        with pytest.raises(RuntimeError):
            with lock.held():
                pass


def test_threads_of_one_process_are_serialized(tmp_path: Path) -> None:
    lock = StoreLock(tmp_path)
    inside = threading.Event()
    leave = threading.Event()
    seen: dict[str, bool] = {}

    def hold() -> None:
        with lock.held():
            inside.set()
            leave.wait(10)

    t = threading.Thread(target=hold)
    t.start()
    assert inside.wait(10)
    with lock.try_held() as got:
        seen["while_held"] = got
    leave.set()
    t.join(10)
    with lock.try_held() as got:
        seen["after"] = got
    assert seen == {"while_held": False, "after": True}


def test_the_current_process_is_alive_and_a_finished_child_is_not() -> None:
    me = ProcessOwner.current()
    assert me.alive()
    with CTX.Manager() as manager:
        out = manager.dict()
        p = CTX.Process(target=lock_workers.record_owner, args=(out,))
        p.start()
        p.join(20)
        dead = ProcessOwner.model_validate(dict(out["owner"]))
    assert dead.pid != me.pid
    assert not dead.alive()


def test_a_different_start_token_on_a_live_pid_is_not_alive() -> None:
    me = ProcessOwner.current()
    assert not ProcessOwner(pid=me.pid, start_token="another-incarnation").alive()


def test_alias_roots_share_one_lock(tmp_path):
    one, two = StoreLock(tmp_path), StoreLock(tmp_path / "child" / "..")
    assert one is two
    with one.held():
        two.assert_held()
        with pytest.raises(RuntimeError):
            with two.held():
                pass
    with pytest.raises(AssertionError):
        two.assert_held()


def test_lock_is_released_on_process_death(tmp_path):
    holding, release = CTX.Event(), CTX.Event()
    p = CTX.Process(target=lock_workers.holder, args=(str(tmp_path), holding, release))
    p.start()
    assert holding.wait(20)
    p.kill()
    p.join(20)
    with StoreLock(tmp_path).try_held() as acquired:
        assert acquired


def test_foreign_host_is_refused_and_old_boot_is_dead(tmp_path):
    import json
    (tmp_path / ".lock").write_text(json.dumps({"host_id": "foreign-host", "boot_id": "old"}))
    with pytest.raises(RuntimeError, match="another host"):
        with StoreLock(tmp_path).held():
            pass
    me = ProcessOwner.current()
    assert not me.model_copy(update={"boot_id": "old-boot"}).alive()
    with pytest.raises(RuntimeError, match="another host"):
        me.model_copy(update={"host_id": "foreign-host"}).alive()


def test_exception_releases_the_lock(tmp_path):
    lock = StoreLock(tmp_path)
    with pytest.raises(ValueError, match="interrupted"):
        with lock.held():
            raise ValueError("interrupted")
    with lock.try_held() as acquired:
        assert acquired


def test_only_the_holding_thread_can_call_locked_helpers(tmp_path):
    lock = StoreLock(tmp_path)
    with lock.held():
        def check():
            with pytest.raises(AssertionError):
                lock.assert_held()
            return True

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(check).result(timeout=10)


def test_process_identity_reads_do_not_spawn_commands(monkeypatch):
    import subprocess

    def refuse(*args, **kwargs):
        pytest.fail("process identity must use the OS directly")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    assert ProcessOwner.current().alive()


def test_unreadable_process_identity_never_allows_takeover(monkeypatch):
    from phaseeval.kernel import lock
    me = ProcessOwner.current()

    start_token = lock._start_token

    def denied(pid):
        if pid == me.pid:
            raise PermissionError(pid)
        return start_token(pid)

    monkeypatch.setattr(lock, "_start_token", denied)
    assert me.alive()


@pytest.mark.parametrize("filesystem", ["nfs", "cifs", "fuse.sshfs", "unknown"])
def test_network_and_unverified_linux_filesystems_are_refused(tmp_path, monkeypatch, filesystem):
    import sys
    read_text = Path.read_text

    def mountinfo(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return f"1 0 0:1 / / rw - {filesystem} source rw\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", mountinfo)
    with pytest.raises(RuntimeError, match="local filesystem"):
        with StoreLock(tmp_path).held():
            pass


def test_unsupported_platform_is_refused(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "platform", "unsupported")
    with pytest.raises(OSError, match="Unsupported"):
        with StoreLock(tmp_path).held():
            pass


def test_windows_boot_identity_uses_a_supported_buffer_size(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from phaseeval.kernel import lock

    def query(kind, buffer, size, needed):
        assert kind == 3
        if size > 48:
            return 0xC0000004
        ctypes.c_int64.from_buffer(buffer).value = 123456
        return 0

    monkeypatch.setattr(lock.sys, "platform", "win32")
    monkeypatch.setattr(lock, "_start_token", lambda pid: "start")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: SimpleNamespace(NtQuerySystemInformation=query), raising=False)
    me = ProcessOwner.current()
    assert me.boot_id == "123456"
    assert me.alive()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_forked_child_neither_owns_nor_releases_the_parent_lock(tmp_path):
    import json

    lock = StoreLock(tmp_path)
    read_fd, write_fd = os.pipe()
    child = None
    try:
        with lock.held():
            child = os.fork()
            if child == 0:
                os.close(read_fd)
                try:
                    lock.assert_held()
                    inherited = True
                except AssertionError:
                    inherited = False
            else:
                os.close(write_fd)
                status = os.waitpid(child, 0)[1]
                assert os.waitstatus_to_exitcode(status) == 0
                results = json.loads(os.read(read_fd, 4096))
                with CTX.Manager() as manager:
                    out = manager.dict()
                    contender = CTX.Process(target=lock_workers.try_once, args=(str(tmp_path), out))
                    contender.start()
                    contender.join(20)
                    assert contender.exitcode == 0
                    results["independent"] = out["acquired"]
                lock.assert_held()
                assert results == {"inherited": False, "reused": False, "independent": False}
        if child == 0:
            with lock.try_held() as acquired:
                os.write(write_fd, json.dumps({"inherited": inherited, "reused": acquired}).encode())
    finally:
        if child == 0:
            os._exit(0)
        os.close(read_fd)
    with lock.try_held() as acquired:
        assert acquired


@pytest.mark.parametrize("name", ["plain", "with space", "with\ttab", "with\nnewline", r"with\backslash", r"literal\040"])
def test_nested_network_mounts_with_escaped_paths_are_refused(tmp_path, monkeypatch, name):
    import sys

    root = tmp_path / name / "store"
    mount = str(root.parent).replace("\\", r"\134").replace(" ", r"\040").replace("\t", r"\011").replace("\n", r"\012")
    read_text = Path.read_text

    def mountinfo(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return f"1 0 0:1 / / rw - ext4 source rw\n2 1 0:2 / {mount} rw - nfs source rw\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", mountinfo)
    with pytest.raises(RuntimeError, match="local filesystem"):
        with StoreLock(root).held():
            pass
