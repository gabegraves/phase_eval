from __future__ import annotations

from pathlib import Path

from phaseeval.kernel.lock import ProcessOwner, StoreLock


def holder(root: str, holding, release) -> None:
    lock = StoreLock(Path(root))
    with lock.held():
        holding.set()
        release.wait(20)


def contender(root: str, holding, tried, results) -> None:
    lock = StoreLock(Path(root))
    holding.wait(20)
    with lock.try_held() as got:
        results["first_try"] = got
    tried.set()
    with lock.held():
        results["eventually"] = True


def record_owner(out) -> None:
    out["owner"] = ProcessOwner.current().model_dump()


def try_once(root: str, out) -> None:
    with StoreLock(Path(root)).try_held() as acquired:
        out["acquired"] = acquired
