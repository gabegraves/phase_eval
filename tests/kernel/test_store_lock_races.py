from __future__ import annotations

import multiprocessing
import os

import pytest

from tests.kernel.concurrency import model

CTX = multiprocessing.get_context("spawn")
SLOT = "slot-1"
REQ = "req-a"


@pytest.fixture(scope="module")
def manager():
    with CTX.Manager() as m:
        yield m


def _world(manager, root) -> dict:
    return {
        "log": manager.list(),
        "claims": manager.dict(),
        "commits": manager.dict(),
        "executions": manager.list(),
        "registry": manager.dict(),
        "root": str(root),
    }


def _run(*procs) -> None:
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]


def _label_race(manager, root, locked: bool):
    world = _world(manager, root)
    results = manager.dict()
    choreography = {"label_read": CTX.Event(), "resume_label": CTX.Event()}
    a = CTX.Process(target=model.label_worker, args=(world, locked, "rater-1", choreography, results))
    b = CTX.Process(target=model.signoff_worker, args=(world, locked, "rater-1", choreography, results))
    _run(a, b)
    return model.Protocol(world, locked), dict(results)


def test_without_the_lock_a_blind_label_lands_after_the_sign_off(manager, tmp_path) -> None:
    proto, results = _label_race(manager, tmp_path, locked=False)
    assert results["label"] == "blind"
    assert [e[0] for e in list(proto.log)] == ["signoff", "label"]
    assert proto.stale_blind_labels() == [1]


def test_with_the_lock_no_blind_label_can_land_after_the_sign_off(manager, tmp_path) -> None:
    proto, results = _label_race(manager, tmp_path, locked=True)
    kinds = [e[0] for e in list(proto.log)]
    assert kinds == ["label", "signoff"], kinds
    assert results["label"] == "blind"
    assert proto.stale_blind_labels() == []


def _claim_race(manager, root, locked: bool):
    world = _world(manager, root)
    results = manager.dict()
    choreography = {"first_checked": CTX.Event(), "resume_first": CTX.Event()}
    a = CTX.Process(target=model.claim_worker, args=(world, locked, SLOT, REQ, "first", choreography, results))
    b = CTX.Process(target=model.claim_worker, args=(world, locked, SLOT, REQ, "second", choreography, results))
    _run(a, b)
    return model.Protocol(world, locked), dict(results)


def test_without_the_lock_two_requests_with_one_key_execute_twice(manager, tmp_path) -> None:
    proto, results = _claim_race(manager, tmp_path, locked=False)
    assert len(list(proto.executions)) == 2, list(proto.executions)
    assert results["second"] == "executed"
    assert results["first"] in ("already", "fenced")


def test_with_the_lock_one_request_executes_and_the_other_is_answered(manager, tmp_path) -> None:
    proto, results = _claim_race(manager, tmp_path, locked=True)
    assert len(list(proto.executions)) == 1, list(proto.executions)
    assert results["first"] == "executed"
    assert results["second"] in ("busy", "already")
    assert len(proto.commits) == 1
    chain = list(proto.claims[SLOT])
    assert [c["generation"] for c in chain] == [0]


def test_a_different_request_on_a_bound_slot_is_a_conflict(manager, tmp_path) -> None:
    proto = model.Protocol(_world(manager, tmp_path), locked=True)
    me = proto.register_self()
    assert proto.claim(SLOT, REQ, me, lambda: None) == "executed"
    assert proto.claim(SLOT, "req-b", me, lambda: None) == "conflict"
    assert proto.claim(SLOT, REQ, me, lambda: None) == "already"
    assert len(list(proto.executions)) == 1


def test_a_dead_owners_claim_is_taken_over_at_the_next_generation(manager, tmp_path) -> None:
    world = _world(manager, tmp_path)
    results = manager.dict()
    _run(CTX.Process(target=model.register_and_exit, args=(world, results)))
    dead = tuple(results["dead"])
    proto = model.Protocol(world, locked=True)
    assert not proto.alive(dead)
    proto.claims[SLOT] = [{"generation": 0, "request": REQ, "owner": dead}]
    me = proto.register_self()
    assert proto.claim(SLOT, REQ, me, lambda: None) == "executed"
    assert [c["generation"] for c in proto.claims[SLOT]] == [0, 1]
    assert proto.commits[SLOT]["generation"] == 1


def test_a_live_owner_with_a_different_start_token_is_dead(manager, tmp_path) -> None:
    proto = model.Protocol(_world(manager, tmp_path), locked=True)
    me = proto.register_self("now")
    stale = (os.getpid(), "before-the-restart")
    assert not proto.alive(stale)
    proto.claims[SLOT] = [{"generation": 0, "request": REQ, "owner": stale}]
    assert proto.claim(SLOT, REQ, me, lambda: None) == "executed"
    assert proto.commits[SLOT]["generation"] == 1


def test_a_live_owners_claim_is_busy_not_taken_over(manager, tmp_path) -> None:
    proto = model.Protocol(_world(manager, tmp_path), locked=True)
    me = proto.register_self()
    proto.claims[SLOT] = [{"generation": 0, "request": REQ, "owner": me.as_tuple()}]
    assert proto.claim(SLOT, REQ, me, lambda: None) == "busy"
    assert list(proto.executions) == []


def test_a_taken_over_owner_cannot_commit(manager, tmp_path) -> None:
    proto = model.Protocol(_world(manager, tmp_path), locked=True)
    me = proto.register_self()

    def take_over():
        with proto.section():
            chain = list(proto.claims[SLOT])
            chain.append({"generation": 1, "request": REQ, "owner": (999999, "other")})
            proto.claims[SLOT] = chain

    assert proto.claim(SLOT, REQ, me, lambda: None, execute=take_over) == "fenced"
    assert SLOT not in proto.commits
