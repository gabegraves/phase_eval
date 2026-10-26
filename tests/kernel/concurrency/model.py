from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phaseeval.kernel.lock import StoreLock


@dataclass(frozen=True)
class Owner:
    pid: int
    token: str

    def as_tuple(self) -> tuple[int, str]:
        return (self.pid, self.token)


def pid_exists(pid: int) -> bool:
    if sys.platform == "win32":
        from phaseeval.kernel.lock import ProcessOwner

        try:
            return ProcessOwner.for_pid(pid).alive()
        except (ValueError, ProcessLookupError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Protocol:

    def __init__(self, world: dict[str, Any], locked: bool) -> None:
        self.log = world["log"]
        self.claims = world["claims"]
        self.commits = world["commits"]
        self.executions = world["executions"]
        self.registry = world["registry"]
        self.lock = StoreLock(Path(world["root"]))
        self.locked = locked

    def register_self(self, token: str | None = None) -> Owner:
        owner = Owner(os.getpid(), token or f"{time.time_ns()}")
        self.registry[owner.pid] = owner.token
        return owner

    def alive(self, owner: tuple[int, str]) -> bool:
        pid, token = owner
        return pid_exists(pid) and self.registry.get(pid) == token

    @contextlib.contextmanager
    def section(self):
        if self.locked:
            with self.lock.held():
                yield
        else:
            yield

    def exposure_of(self, rater: str) -> str:
        return "revealed" if any(e[0] == "signoff" and e[1] == rater for e in list(self.log)) else "blind"

    def label(self, rater: str, pause: Callable[[], None]) -> str:
        with self.section():
            exposure = self.exposure_of(rater)
            pause()
            self.log.append(("label", rater, exposure))
        return exposure

    def sign_off(self, rater: str) -> None:
        with self.section():
            self.log.append(("signoff", rater))

    def stale_blind_labels(self) -> list[int]:
        entries = list(self.log)
        bad = []
        for i, e in enumerate(entries):
            if e[0] == "label" and e[2] == "blind":
                if any(p[0] == "signoff" and p[1] == e[1] for p in entries[:i]):
                    bad.append(i)
        return bad

    def claim(self, slot: str, request: str, me: Owner, pause: Callable[[], None], *, execute=lambda: None) -> str:
        with self.section():
            chain = list(self.claims.get(slot, []))
            if chain and chain[0]["request"] != request:
                return "conflict"
            if slot in self.commits:
                return "already" if self.commits[slot]["request"] == request else "conflict"
            latest = chain[-1] if chain else None
            if latest is not None and self.alive(latest["owner"]):
                return "busy"
            pause()
            generation = latest["generation"] + 1 if latest else 0
            chain.append({"generation": generation, "request": request, "owner": me.as_tuple()})
            self.claims[slot] = chain
        execute()
        self.executions.append((me.pid, generation))
        with self.section():
            latest = list(self.claims[slot])[-1]
            if latest["generation"] != generation or tuple(latest["owner"]) != me.as_tuple():
                return "fenced"
            if slot in self.commits:
                return "already" if self.commits[slot]["request"] == request else "conflict"
            self.commits[slot] = {"generation": generation, "request": request, "child": f"child-{me.pid}-{generation}"}
        return "executed"


def label_worker(world: dict[str, Any], locked: bool, rater: str, choreography: dict[str, Any], results: Any) -> None:
    proto = Protocol(world, locked)
    proto.register_self()

    def pause() -> None:
        if "label_read" in choreography:
            choreography["label_read"].set()
        if "resume_label" in choreography:
            choreography["resume_label"].wait(10)

    results["label"] = proto.label(rater, pause)


def signoff_worker(world: dict[str, Any], locked: bool, rater: str, choreography: dict[str, Any], results: Any) -> None:
    proto = Protocol(world, locked)
    proto.register_self()
    choreography["label_read"].wait(10)
    if locked:
        choreography["resume_label"].set()
    proto.sign_off(rater)
    if not locked:
        choreography["resume_label"].set()
    results["signoff"] = "done"


def claim_worker(world: dict[str, Any], locked: bool, slot: str, request: str, role: str, choreography: dict[str, Any], results: Any) -> None:
    proto = Protocol(world, locked)
    me = proto.register_self()

    def pause() -> None:
        if role == "first":
            choreography["first_checked"].set()
            choreography["resume_first"].wait(10)

    if role == "second":
        choreography["first_checked"].wait(10)
        if locked:
            choreography["resume_first"].set()
    results[role] = proto.claim(slot, request, me, pause)
    if role == "second" and not locked:
        choreography["resume_first"].set()


def register_and_exit(world: dict[str, Any], results: Any) -> None:
    proto = Protocol(world, locked=True)
    owner = proto.register_self("gone")
    results["dead"] = owner.as_tuple()
