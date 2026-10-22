"""U4 view: snapshots of imported DEV006 answers, the blind review payload and its reveal.

One real store in a temporary directory: the frozen DEV006 package as a root and two harness v2
trials imported through the importer, a qwen answer (the three precursors) and an Astra answer
(the precursors plus Bi2CO5, CIF001772). The routes are driven through the served app.

The U6 runs are git-excluded and exist only on the rig. From another worktree, point
AXA_WORKBENCH_CUSTODY at a data/custody/workbench directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from phaseeval.api.app import app
from phaseeval.api.auth import CaseOperation, Principal, PrincipalAuthenticator, PrincipalType
from phaseeval.workbench import importer
from phaseeval.workbench import snapshot as snapshot_module
from phaseeval.workbench.executors import default_registry
from phaseeval.workbench.labels import ExposureLedger, LabelStore, PermissionRequest, SignOffRequest
from phaseeval.workbench.model import ChildAttempt, ImportRun, digest_of
from phaseeval.workbench.page_routes import ReviewQueue, queue_path
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.snapshot import Dia, attempt_payload, build_snapshot, k2
from phaseeval.workbench.store import ActionRefused, WorkbenchStore

WORKTREE = Path(__file__).resolve().parents[2]
CASES = WORKTREE / "data/workbench/cases"
LABELS = WORKTREE / "data/workbench/case_labels.json"
CUSTODY = Path(os.environ.get("AXA_WORKBENCH_CUSTODY", WORKTREE / "data/custody/workbench"))
JOBS = CUSTODY / "runs/jobs"
QWEN = JOBS / "u6-dev006-qwen-006/task-dev006__5SKiLMp"
ASTRA = JOBS / "u6-dev006-astra-006/task-dev006__eTpQJSY"
# Frozen after the assignment, as an export would: outside the review cohort (D12).
LATE = JOBS / "u6-dev006-qwen-007/task-dev006__eWndMtv"
BI2CO5 = "CIF001772"
# The U16 smoke store: the owner added Bi2CO5 to the series-004 qwen import and refit (8 threads).
U16_STORE = CUSTODY / "u16-smoke/20261021T172541Z/store"
U16_REFIT = "8167c0f81249043c8bade4d08155cd4c943e1a0527ae30f30f883b807d053ab1"
U16_OWNER = "u16-smoke-owner"

pytestmark = pytest.mark.skipif(
    not (CUSTODY / "DEV006/custody.json").is_file()
    or not all(trial.is_dir() for trial in (QWEN, ASTRA, LATE)),
    reason=f"custody records or U6 runs absent under {CUSTODY}; set AXA_WORKBENCH_CUSTODY",
)

OWNER = Principal(
    "owner", PrincipalType.USER, frozenset({"owner"})
)  # admin: loads, grants, assigns
RATER = Principal("rater-1", PrincipalType.USER, frozenset({"rater"}))  # label mode (D11)
IMPORTER = Principal("importer", PrincipalType.SERVICE, frozenset({"importer"}))
TOKEN = "test-token"

# PI security directive: nothing before sign-off names the run, the reference or a sibling.
BANNED_KEYS = {"job", "job_name", "model", "model_id", "harness", "harness_version", "series"}
BANNED_KEYS |= {"trial_id", "reference_match", "reference", "episode_id", "siblings"}
# A source path names the run too: a job directory spells the model and series.
BANNED_TEXT = re.compile(
    r"u6-|harbor-local|harbor-live|harbor-at|data/custody|series-\d|/jobs/|task-dev\d+__|qwen|astra|gpt-6",
    re.IGNORECASE,
)


def write_principals(path: Path, principal: Principal) -> Path:
    registry = {
        "schema_version": "axa.api_principals.v1",
        "principals": [
            {
                "principal_id": principal.principal_id,
                "principal_type": "user",
                "credential_sha256": hashlib.sha256(TOKEN.encode()).hexdigest(),
                "roles": sorted(principal.roles),
            }
        ],
    }
    path.write_text(json.dumps(registry))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict:
    base = tmp_path_factory.mktemp("view")
    store = WorkbenchStore(
        base / "store",
        default_registry().run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=CUSTODY / "runs",
        exposure=ExposureLedger(base / "store"),
    )
    case, blobs = importer.read_case_package(
        CASES / "DEV006", LABELS, custody=CUSTODY / "DEV006/custody.json"
    )
    root = store.load_case(OWNER, case, blobs)
    origins = {}
    for name, trial in (("qwen", QWEN), ("astra", ASTRA), ("late", LATE)):
        run, result, run_blobs = importer.read_harbor_run(trial, case)
        origins[name] = store.import_run(IMPORTER, root, run, result, run_blobs)
    labels = LabelStore(store)
    snaps = {
        name: build_snapshot(store, OWNER, origin, "agent_answer")
        for name, origin in origins.items()
        if name != "late"
    }
    digests = {name: labels.put_snapshot(snap) for name, snap in snaps.items()}
    store.grant(OWNER, root, RATER.principal_id, (CaseOperation.INSPECT,))
    labels.assign(OWNER, root, (RATER.principal_id,), tuple(sorted(digests.values())))
    labels.put_snapshot(build_snapshot(store, OWNER, origins["late"], "agent_answer"))
    queue = ReviewQueue(principal_id=RATER.principal_id, snapshots=tuple(sorted(digests.values())))
    queue_path(store, RATER.principal_id).parent.mkdir()
    queue_path(store, RATER.principal_id).write_text(queue.model_dump_json())
    principals = write_principals(base / "principals.json", RATER)
    return {
        "store": store,
        "labels": labels,
        "root": root,
        "origins": origins,
        "snaps": snaps,
        "digests": digests,
        "principals": principals,
    }


def serve(store: WorkbenchStore, principals: Path) -> Iterator[TestClient]:
    saved = dict(app.state._state)
    app.state.workbench_store = store
    app.state.attempt_authenticator = PrincipalAuthenticator.from_file(principals)
    with TestClient(app) as test_client:
        test_client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield test_client
    app.state._state.clear()
    app.state._state.update(saved)


@pytest.fixture()
def client(world: dict, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("AXA_WORKBENCH_CUSTODY", str(CUSTODY))
    yield from serve(world["store"], world["principals"])


def leaks(value: object, path: str = "$") -> Iterator[str]:
    """Every banned key, at any depth, and every string that names a source path or model."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in BANNED_KEYS:
                yield f"{path}.{key}"
            yield from leaks(key, f"{path}.<key>")
            yield from leaks(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaks(item, f"{path}[{index}]")
    elif isinstance(value, str) and BANNED_TEXT.search(value):
        yield f"{path}: {value[:120]!r}"


def test_the_qwen_answer_lists_the_residual_peak_near_23_95_degrees(world: dict) -> None:
    snap = world["snaps"]["qwen"]
    assert [p.phase_id for p in snap.phases] == ["CIF012075", "COD2310703", "CIF001783"]
    (peak,) = snap.residual_peaks
    assert abs(peak.two_theta_deg - 23.95) < 0.05 and peak.peak_id == "p23.95"
    assert 2.0 <= peak.height_percent < 5.0  # a weak unassigned peak: listed, not flagged (K2)
    checks = {c.code: c for c in snap.checks}
    assert {c.policy_version for c in snap.checks} == {"kchecks-2026-10-21-p1"}
    assert not hasattr(snapshot_module, "K2_FLAG_PERCENT")  # thresholds live once, in triage (V1)
    assert checks["K2"].status == "pass" and "23.951 deg" in checks["K2"].detail
    assert checks["K5"].status == "pass"
    assert (
        snap.answered
        and snap.stage == "rietveld"
        and snap.rwp_percent == pytest.approx(3.54, abs=0.01)
    )


def test_the_astra_answer_with_bi2co5_leaves_no_residual_peak(world: dict) -> None:
    snap = world["snaps"]["astra"]
    assert BI2CO5 in [p.phase_id for p in snap.phases]
    bi2co5 = next(p for p in snap.phases if p.phase_id == BI2CO5)
    assert bi2co5.formula == "Bi2CO5" and bi2co5.weight_percent == pytest.approx(2.74, abs=0.01)
    assert snap.residual_peaks == ()
    assert {c.code: c.status for c in snap.checks}["K2"] == "pass"


def test_build_snapshot_is_deterministic(world: dict) -> None:
    store, labels = world["store"], world["labels"]
    for name in world["snaps"]:
        again = build_snapshot(store, OWNER, world["origins"][name], "agent_answer")
        assert digest_of(again) == digest_of(world["snaps"][name])
        assert labels.put_snapshot(again) == world["digests"][name]


def test_the_snapshot_freezes_chemistry_for_each_answered_phase(world: dict) -> None:
    snap = world["snaps"]["astra"]
    assert snap.chemistry is not None
    assert [p.phase_id for p in snap.chemistry.phases] == [p.phase_id for p in snap.phases]


def test_nothing_the_page_reads_before_sign_off_names_the_run(
    world: dict, client: TestClient
) -> None:
    queue = client.get("/api/attempts/queue")
    assert queue.status_code == 200
    items = queue.json()["items"]
    assert sorted(i["digest"] for i in items) == sorted(world["digests"].values())
    fetched = {"queue": queue.json()}
    for item in items:
        review = client.get(f"/api/attempts/snapshots/{item['digest']}/review")
        assert review.status_code == 200
        fetched[item["digest"]] = review.json()
        assert client.get(f"/api/attempts/snapshots/{item['digest']}/revealed").status_code == 403
    assert [leak for body in fetched.values() for leak in leaks(body)] == []
    qwen = fetched[world["digests"]["qwen"]]
    assert {
        "brief",
        "answer",
        "fit",
        "checks",
        "steps",
        "missing",
        "residual_peaks",
        "chemistry",
    } <= set(qwen)
    assert qwen["fit"]["phase_curves"] and len(qwen["fit"]["two_theta"]) == qwen["fit"]["points"]
    assert all(
        step["attempt"] == world["origins"]["qwen"]
        for step in qwen["steps"]
        if step["kind"] != "compaction"
    )
    assert [s["call_index"] for s in qwen["steps"] if s["kind"] == "refine"] == sorted(
        s["call_index"] for s in qwen["steps"] if s["kind"] == "refine"
    )


def test_the_page_and_the_judge_see_the_answer_as_submitted(
    world: dict, client: TestClient
) -> None:
    """Science review (AXA-43zwk.10): the answer's own phase words and quantity basis, apart
    from the CIF metadata, and native evidence around each listed residual peak."""
    snap = world["snaps"]["qwen"]
    assert [p.reported_formula for p in snap.phases] == ["NiO", "Li2CO3", "Bi2O3"]
    assert isinstance(snap.reported_quantity_basis, str)
    assert snap.reported_quantity_basis.startswith("Relative modeled crystalline weight fractions")
    body = client.get(f"/api/attempts/snapshots/{world['digests']['qwen']}/review").json()
    phase = body["answer"]["phases"][0]
    assert phase["formula"] == "NiO" and phase["cif"] == {
        "formula": "NiO",
        "space_group": "Fm-3m (225)",
    }
    assert body["answer"]["quantity_basis"] == snap.reported_quantity_basis
    assert body["brief"]["precursor_amounts_note"] == "withheld by task policy"
    assert body["answer"]["submitted"] == snap.reported_answer is not None
    assert body["fit"]["native_points"] == 8282 and body["fit"]["stride"] == 7
    assert body["fit"]["precision"] == "4 significant digits"
    (window,) = body["residual_windows"]
    assert window["peak_id"] == "p23.95"
    assert len(window["two_theta_deg"]) > 80  # native 0.0109 deg steps over 1 deg
    assert max(abs(t - 23.951) for t in window["two_theta_deg"]) <= 0.5
    assert set(window["phase_curves"]) == set(body["fit"]["phases"])
    astra = client.get(f"/api/attempts/snapshots/{world['digests']['astra']}/review").json()
    assert astra["residual_windows"] == []


def test_a_phase_that_leaves_no_residual_still_shows_its_own_peaks_at_native_resolution(
    world: dict, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Science (AXA-43zwk.10 08:42): Astra's Bi2CO5 removes the residual, so no residual window
    exists; each reported phase's own calculated peaks still get native windows, chosen from the
    fit alone and shared with the judge."""

    def no_reference(*_: object) -> None:
        raise AssertionError("the blind view read the reference")

    monkeypatch.setattr(snapshot_module, "_reference", no_reference)
    body = client.get(f"/api/attempts/snapshots/{world['digests']['astra']}/review").json()
    assert body["residual_windows"] == []
    reported = [p["candidate_id"] for p in body["answer"]["phases"]]
    assert BI2CO5 in reported
    for cid in reported:
        assert len([w for w in body["phase_windows"] if w["phase_id"] == cid]) == snapshot_module.PHASE_WINDOWS
    rows = snapshot_module.evidence(world["store"], OWNER, world["origins"]["astra"], "agent_answer").dia.rows
    for window in body["phase_windows"]:
        near = rows[np.abs(rows[:, 0] - window["center_deg"]) <= snapshot_module.RESIDUAL_WINDOW_DEG]
        assert window["two_theta_deg"] == near[:, 0].tolist() and window["observed"] == near[:, 1].tolist()
        assert window["calculated"] == near[:, 2].tolist() and window["background"] == near[:, 3].tolist()
        assert set(window["phase_curves"]) == set(reported)
    assert body["phase_window_rule"] == snapshot_module.PHASE_WINDOW_RULE
    assert list(leaks(body)) == []


def test_every_api_route_requires_a_bearer_token(world: dict, client: TestClient) -> None:
    digest = world["digests"]["qwen"]
    for url in ("/api/attempts/queue", f"/api/attempts/snapshots/{digest}/review", f"/api/attempts/snapshots/{digest}/revealed"):  # fmt: skip
        assert client.get(url, headers={"Authorization": ""}).status_code == 401


def test_the_reveal_opens_only_after_this_raters_sign_off(world: dict, client: TestClient) -> None:
    labels, digests = world["labels"], world["digests"]
    url = f"/api/attempts/snapshots/{digests['qwen']}/revealed"
    assert client.get(url).status_code == 403
    labels.record_permission(
        RATER,
        PermissionRequest(
            participation_and_storage=True,
            internal_evaluation_and_calibration=True,
            public_release=False,
            training=False,
            public_identity_disclosure=False,
        ),
    )
    labels.sign_off(RATER, SignOffRequest(snapshot=digests["qwen"], verdict="would_sign"))
    assert client.get(url).status_code == 403  # the Astra answer of this pattern is unsigned
    labels.sign_off(RATER, SignOffRequest(snapshot=digests["astra"], verdict="unresolved"))
    # The late qwen-007 snapshot stays unsigned: the reveal waits on the cohort only (D12).
    revealed = client.get(url)
    assert revealed.status_code == 200
    body = revealed.json()
    assert (
        body["identity"]["job"] == "u6-dev006-qwen-006"
        and body["identity"]["model"] == "qwen36-27b"
    )
    assert body["reference_revealed"] is True
    assert body["reference_match"] == {
        "basis": "answer",
        "rule": "formula",
        "phase_list_exact": True,
        "missed": [],
        "extra": [],
    }
    # D14: the siblings are the rater's cohort, not the late qwen-007 snapshot.
    assert sorted(s["job"] for s in body["siblings"]) == [
        "u6-dev006-astra-006",
        "u6-dev006-qwen-006",
    ]
    assert [s["this"] for s in body["siblings"] if s["job"] == "u6-dev006-qwen-006"] == [True]
    astra = next(s for s in body["siblings"] if s["job"] == "u6-dev006-astra-006")
    assert astra["reference_match"]["extra"] == [BI2CO5]


@pytest.mark.skipif(not U16_STORE.is_dir(), reason=f"no U16 smoke store at {U16_STORE}")
def test_a_refit_child_of_an_imported_run_reviews_blind(tmp_path: Path) -> None:
    copy = tmp_path / "store"
    shutil.copytree(U16_STORE, copy, symlinks=True)
    store = WorkbenchStore(
        copy,
        default_registry().run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=CUSTODY / "runs",
        exposure=ExposureLedger(copy),
    )
    principals = write_principals(
        tmp_path / "principals.json", Principal(U16_OWNER, PrincipalType.USER, frozenset({"owner"}))
    )
    for client in serve(store, principals):
        response = client.get(f"/api/attempts/{U16_REFIT}/review")
        assert response.status_code == 200
        body = response.json()
        assert list(leaks(body)) == []
        assert body["kind"] == "refit" and body["digest"] is None and body["answer"] is None
        assert body["fit"]["rwp"] == pytest.approx(2.50, abs=0.01)
        assert BI2CO5 in body["fit"]["phases"] and body["residual_peaks"] == []
        assert [s["title"] for s in body["steps"]] == [f"Added {BI2CO5} (Bi2CO5)", "Refine"]
        assert (
            client.get(
                f"/api/attempts/{U16_REFIT}/review", headers={"Authorization": ""}
            ).status_code
            == 401
        )


def test_a_residual_peak_at_exactly_the_flag_height_flags() -> None:
    """K2 boundary: a peak whose height is the policy's flag height flags (>=, not >)."""
    tt = np.arange(10.0, 60.0, 0.01)
    background = np.full_like(tt, 1000.0)
    calculated = background + 1.0e5 * np.exp(-0.5 * ((tt - 30.0) / 0.05) ** 2)
    bump = np.exp(-0.5 * ((tt - 45.0) / 0.05) ** 2)

    def dia(amplitude: float) -> Dia:
        observed = calculated + amplitude * bump
        return Dia(None, "rietveld", (), np.column_stack([tt, observed, calculated, background]))

    first = k2(dia(1.0e4))["peaks"][0]["pct_of_strongest_peak"]  # the height scales linearly
    check = k2(dia(1.0e4 * 5.0 / first))
    (peak,) = check["peaks"]
    assert peak["pct_of_strongest_peak"] == 5.0 and peak["status"] == "flag"
    assert check["status"] == "flag"
    assert k2(dia(1.0e4 * 4.99 / first))["status"] == "pass"


# --- the answer file: bound by digest, shown as submitted (AXA-43zwk.21) -------------------------

ASTRA_007 = JOBS / "u6-dev006-astra-007/task-dev006__nrBCdaa"
QWEN_005 = JOBS / "u6-dev006-qwen-005/task-dev006__JVabvY8"
OUTPUT = "artifacts/workspace/output"


def fresh(base: Path, trial: Path, **end: object) -> tuple[WorkbenchStore, str]:
    """A new store holding the DEV006 root and `trial` imported, its end updated by `end`."""
    store = WorkbenchStore(
        base / "store",
        default_registry().run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=CUSTODY / "runs",
        exposure=ExposureLedger(base / "store"),
    )
    case, blobs = importer.read_case_package(
        CASES / "DEV006", LABELS, custody=CUSTODY / "DEV006/custody.json"
    )
    root = store.load_case(OWNER, case, blobs)
    run, result, run_blobs = importer.read_harbor_run(trial, case)
    if end:
        result = result.model_copy(update={"end": result.end.model_copy(update=end)})
    return store, store.import_run(IMPORTER, root, run, result, run_blobs)


def answer_shaped(data: bytes) -> dict | None:
    """A receipt the first-match selection treated as an answer: phases, case_id and weights."""
    try:
        value = json.loads(data)
    except ValueError:
        return None
    shaped = isinstance(value, dict) and {"phases", "case_id", "weight_fractions"} <= set(value)
    return value if shaped else None


def leaves(value: object, path: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}/{index}")
    else:
        yield path, value


def test_drafts_with_the_answers_phases_and_amounts_are_never_shown_as_the_answer(
    tmp_path: Path,
) -> None:
    """Three drafts report the final answer's phases and amounts in other words, and sort before
    the final file in sha256 receipt order, where the first-match selection picked them."""
    trial = tmp_path / "trial"
    shutil.copytree(QWEN, trial, symlinks=True)
    final_bytes = (trial / OUTPUT / "analysis.json").read_bytes()
    final = json.loads(final_bytes)
    for n in range(3):
        nonce = 0
        while True:
            draft = final | {
                "quantity_basis": f"Draft {n} basis, replaced in the final answer ({nonce}).",
                "limitations": [f"Draft {n} limitation that the final answer does not make."],
            }
            data = json.dumps(draft).encode()
            if hashlib.sha256(data).hexdigest() < hashlib.sha256(final_bytes).hexdigest():
                break
            nonce += 1
        (trial / OUTPUT / f"analysis_draft{n}.json").write_bytes(data)
    store, origin = fresh(tmp_path, trial)
    receipts = store._load(origin).outcome.origin.receipts
    first = next(v for r in receipts if (v := answer_shaped(store.read_blob(r.sha256))))
    assert first["quantity_basis"].startswith("Draft")  # the old selection would show a draft

    snap = build_snapshot(store, OWNER, origin, "agent_answer")
    assert snap.reported_quantity_basis == final["quantity_basis"]
    assert not any("Draft" in claim for claim in snap.claims)
    assert snap.reported_answer["limitations"] == final["limitations"]


@pytest.mark.skipif(not ASTRA_007.is_dir(), reason=f"no {ASTRA_007.name} under {JOBS}")
def test_dev006_astra_007_shows_its_final_answer_file_not_its_draft(tmp_path: Path) -> None:
    """Science review 6924a759f: the first match was analysis_provisional_fourphase.json, 11
    claims; the final analysis.json has 26 under the same extractor."""
    final = json.loads((ASTRA_007 / OUTPUT / "analysis.json").read_bytes())
    drafts = [
        draft
        for path in sorted((ASTRA_007 / OUTPUT).rglob("analysis_*.json"))
        if (draft := answer_shaped(path.read_bytes()))
        and [p["candidate_id"] for p in draft["phases"]]
        == [p["candidate_id"] for p in final["phases"]]
        and draft["weight_fractions"] == final["weight_fractions"]
    ]
    assert any(d["quantity_basis"] != final["quantity_basis"] for d in drafts)  # a real trap

    store, origin = fresh(tmp_path, ASTRA_007)
    snap = build_snapshot(store, OWNER, origin, "agent_answer")
    assert len(snap.claims) == 26
    kept = dict(leaves(snap.reported_answer))
    for path, value in leaves(final):
        if path.startswith("/fit_artifacts/"):
            assert path not in kept
        elif isinstance(value, str) and "/workspace/" in value:
            assert "/workspace/" not in kept[path] and "[path]" in kept[path]
        else:
            assert kept[path] == value, path  # every other leaf, verbatim
    assert set(kept) == {p for p, _ in leaves(final) if not p.startswith("/fit_artifacts/")}
    payload = attempt_payload(store, OWNER, origin)
    assert payload["answer"]["submitted"] == snap.reported_answer
    assert list(leaks(payload)) == []


@pytest.mark.skipif(not QWEN_005.is_dir(), reason=f"no {QWEN_005.name} under {JOBS}")
def test_the_answer_as_submitted_keeps_its_fit_statistics_numbers_and_units(tmp_path: Path) -> None:
    store, origin = fresh(tmp_path, QWEN_005)
    snap = build_snapshot(store, OWNER, origin, "agent_answer")
    assert snap.reported_answer["fit_statistics"]["residual_statistics_attempt2"] == {
        "max_absolute_difference": {"value": 891.7, "unit": "counts"},
        "RMS_difference": {"value": 120.45, "unit": "counts"},
        "mean_absolute_difference": {"value": 76.2, "unit": "counts"},
        "fraction_points_above_1pct_max_obs": {"value": 7.4, "unit": "percent"},
    }
    final = json.loads((QWEN_005 / OUTPUT / "analysis.json").read_bytes())
    # Its only paths are its fit_artifacts, so the rest is the file as written.
    assert snap.reported_answer == {k: v for k, v in final.items() if k != "fit_artifacts"}
    assert list(leaks(snap.model_dump(mode="json"))) == []


@pytest.mark.skipif(not ASTRA_007.is_dir(), reason=f"no {ASTRA_007.name} under {JOBS}")
@pytest.mark.parametrize("binding", ["none", "not_the_answer"])
def test_an_import_bound_to_no_answer_file_shows_no_answer_words(
    tmp_path: Path, binding: str
) -> None:
    """Unbound, as every import written before the binding, or bound to a receipt that is not
    this answer: the phases and amounts stand, the words are unavailable, and no draft (this
    trial has drafts with the same amounts) stands in."""
    config = hashlib.sha256((ASTRA_007 / "config.json").read_bytes()).hexdigest()
    store, origin = fresh(tmp_path, ASTRA_007, answer_sha256=None if binding == "none" else config)
    snap = build_snapshot(store, OWNER, origin, "agent_answer")
    final = json.loads((ASTRA_007 / OUTPUT / "analysis.json").read_bytes())
    assert snap.answered and [p.phase_id for p in snap.phases] == [
        p["candidate_id"] for p in final["phases"]
    ]
    assert (snap.claims, snap.reported_quantity_basis, snap.reported_answer) == ((), None, None)
    assert all(p.reported_formula is None for p in snap.phases)
    payload = attempt_payload(store, OWNER, origin)
    assert "answer text" in payload["missing"] and payload["answer"]["submitted"] is None


def test_an_import_naming_an_answer_file_outside_its_receipts_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ActionRefused, match="not one of its receipts"):
        fresh(tmp_path, QWEN, answer_sha256="0" * 64)


@pytest.mark.skipif(not U16_STORE.is_dir(), reason=f"no U16 smoke store at {U16_STORE}")
def test_an_import_written_before_the_binding_loads_with_its_words_unavailable(
    tmp_path: Path,
) -> None:
    copy = tmp_path / "store"
    shutil.copytree(U16_STORE, copy, symlinks=True)
    store = WorkbenchStore(
        copy,
        default_registry().run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=CUSTODY / "runs",
        exposure=ExposureLedger(copy),
    )
    owner = Principal(U16_OWNER, PrincipalType.USER, frozenset({"owner"}))
    digest, attempt = next(
        (d, a)
        for d, a in store.lineage(owner, U16_REFIT)
        if isinstance(a, ChildAttempt) and isinstance(a.action, ImportRun)
    )
    assert attempt.outcome.result.end.answer_sha256 is None  # the old importer wrote it
    snap = build_snapshot(store, owner, digest, "agent_answer")
    assert snap.answered and snap.phases
    assert (snap.claims, snap.reported_quantity_basis, snap.reported_answer) == ((), None, None)


def test_the_scrub_replaces_paths_and_run_names_and_keeps_formulas_ratios_and_numbers() -> None:
    names = ("u6-dev006-astra-007", "task-dev006__nrBCdaa", "gpt-6-astra", "astra", "codex")
    value = {
        "evidence": "P2_1/c, Rwp/Rexp = 1.8; see /workspace/output/run1/result.json, https://doi.org/10.1107/x",
        "result_file": "/workspace/output/h1_search/result.json",
        "note": "Astra's run task-dev006__nrBCdaa in codex; Astragalus and gpt-6-astra-like stay apart",
        "values": [891.7, None, True, 0, "counts"],
    }
    assert snapshot_module._scrubbed(value, names) == {
        "evidence": "P2_1/c, Rwp/Rexp = 1.8; see [path], https://doi.org/10.1107/x",
        "result_file": "[path]",
        "note": "[run identity]'s run [run identity] in [run identity]; Astragalus and [run identity]-like stay apart",
        "values": [891.7, None, True, 0, "counts"],
    }


@pytest.mark.parametrize("case_id", ["DEV006", "DEV102"])
def test_the_brief_says_withheld_amounts_are_policy_and_keeps_recorded_masses(case_id: str) -> None:
    """Science (AXA-43zwk.8 09:54 UTC): the package's withheld notice explains its policy with
    the reference fractions, which the brief must not present as sample composition. The
    package bytes stay as frozen; the page and the judge read this one brief."""
    case, blobs = importer.read_case_package(
        CASES / case_id, LABELS, custody=CUSTODY / case_id / "custody.json"
    )
    missing: list[str] = []
    brief = snapshot_module._brief(SimpleNamespace(read_blob=blobs.__getitem__), case, missing)
    package = json.loads((CASES / case_id / "manifest.json").read_bytes())["cases"][0]
    masses = [p.get("actual_mass_g") for p in package["material_context"]["precursors"]]
    assert [p["mass_g"] for p in brief["precursors"]] == masses
    if case_id == "DEV006":
        assert package["material_context"]["precursor_amounts"].endswith(
            "reference weight fractions"
        )
        assert brief["precursor_amounts_note"] == "withheld by task policy"
        assert "reference weight fractions" not in json.dumps(brief)
        assert "brief.precursors[].mass_g" in missing
    else:
        assert masses == [1.38156, 0.62231] and brief["precursor_amounts_note"] is None
        assert "brief.precursors[].mass_g" not in missing
        assert [s["action"] for s in brief["processing"]][:2] == ["PowderDosing", "Heating"]
