"""U6 episode export and the eval dataset: schema, exclusions, completeness, label separation,
custody and withdrawal (R11, R18, R19, R33; overnight N4, N8)."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_tree import ADD, CASE, IMPORTER, OWNER, REFINE, RUN_START, World

from phaseeval.api.auth import CaseOperation, Principal, PrincipalType
from phaseeval.workbench import cli, exports, importer, policy, snapshot
from phaseeval.workbench.executors import default_registry
from phaseeval.workbench.exports import (
    ExportInputs,
    RaterPseudonyms,
    build_dataset,
    dataset_json_schemas,
    episode_json_schema,
    export_episodes,
    playback_states,
    validate_dataset,
    write_dataset,
)
from phaseeval.workbench.labels import (
    ExposureLedger,
    JudgeIdentity,
    Label,
    LabelRequest,
    LabelStore,
    PeakCallRequest,
    PermissionRequest,
    Rater,
    SignOffRequest,
    StepFeedbackRequest,
    StepRef,
)
from phaseeval.workbench.model import BlobDigest, ImportedRunResult, ImportRun
from phaseeval.workbench.policy import EligibilityLedger, RightsRegistry
from phaseeval.workbench.qualification import Producer, QualificationPolicy, Qualifier
from phaseeval.workbench.seams import NoReference, public_seams
from phaseeval.workbench.snapshot import AnalysisSnapshot, ResidualPeak, SnapshotPhase
from phaseeval.workbench.store import WorkbenchStore

WORKTREE = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 21, 17, tzinfo=UTC)
AGENT = Principal("agent-1", PrincipalType.AGENT, frozenset())
RATER = Principal("rater-ada", PrincipalType.USER, frozenset({"rater"}))
OWNER_RATER = Principal("owner-rater", PrincipalType.USER, frozenset({"owner_rater"}))  # label mode (D11)
CURVE = b"two_theta,observed,calculated,difference\n10.00,120,118,2\n10.02,131,130,1\n"
FORMULAS = {c.candidate_id: c.formula_sum for c in CASE.candidates}
INPUTS = ExportInputs(
    dataset_version="dataset-test",
    eligibility=EligibilityLedger(policy_version="policy-test", decisions={}),
    rights=RightsRegistry(version="rights-test", artifacts={}, contributors={}, model_outputs={}),
    now=NOW,
)


def _ref(data: bytes, media_type: str, role: str) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "media_type": media_type, "role": role}


BASIS_FIT = {
    "stage": "rietveld",
    "quantification": "restricted_profile",
    "rwp_percent": 4.1,
    "gof": 1.8,
    "phases": [
        {"candidate_id": cid, "weight_percent": w, "weight_percent_error": None, "cell": dict.fromkeys("a b c alpha beta gamma".split())}
        for cid, w in (("CIF000001", 60.0), ("CIF000002", 40.0))
    ],
    "curve": _ref(CURVE, "text/csv", "fit_curve"),
}
ANSWERED = {"end": "submitted", "answer": {"verdict": "NOT_RECORDED", "phases": ["CIF000001", "CIF000002"]}}


def import_run(world: World, trial: str, *, end: dict, completeness: dict, basis_fit: dict | None) -> str:
    receipt = json.dumps({"trial": trial}).encode()
    action = ImportRun.model_validate(
        {
            "harbor": {
                "job_name": "u6-dev900-qwen-006",
                "trial_id": trial,
                "task_sha256": "4" * 64,
                "pattern_sha256": CASE.pattern.sha256,
                "cif_sha256": {c.candidate_id: c.cif.sha256 for c in CASE.candidates[:2]},
            }
        }
    )
    result = ImportedRunResult.model_validate(
        {
            "agent": {"model_id": "qwen36-27b", "model_version": None, "harness": "harbor-local-mcp", "harness_version": "2"},
            "budget_wall_seconds": 3440,
            "wall_seconds": 900.0,
            "cost_usd": None,
            "end": end,
            "basis_fit": basis_fit,
            "trajectory": [{"index": 0, "tool": "refine", "arguments_sha256": "6" * 64, "summary": "refine CIF000001,CIF000002"}],
            "completeness": completeness,
        }
    )
    blobs = {BlobDigest(hashlib.sha256(data).hexdigest()): data for data in (receipt, CURVE)}
    return world.store.import_run(IMPORTER, world.root, action, result, blobs)


class Tree:
    """An answered import corrected by the owner, an import missing its basis fit, an
    unsuccessful import, and a host-side agent session."""

    def __init__(self, root: Path) -> None:
        self.world = World(root)
        world = self.world
        self.answered = import_run(world, "task-dev900__a", end=ANSWERED, completeness={"kind": "complete"}, basis_fit=BASIS_FIT)
        # The owner has seen this dev case's reference (LB2: exposure comes from the ledger).
        LabelStore(world.store).declare_exposure(OWNER, OWNER.principal_id, world.root, "dev case reference seen")
        added = world.act(OWNER, self.answered, "add-3", {"verb": "add_phases", "candidates": ["CIF000003"]})
        self.corrected = world.act(OWNER, added.digest, "refit-3", REFINE).digest
        self.incomplete = import_run(
            world,
            "task-dev900__b",
            end=ANSWERED,
            completeness={"kind": "incomplete", "missing": ["basis fit: output/run1/run1.lst"]},
            basis_fit=None,
        )
        self.unsuccessful = import_run(
            world,
            "task-dev900__c",
            end={"end": "unsuccessful", "reason": "budget_exhausted"},
            completeness={"kind": "complete"},
            basis_fit=None,
        )
        world.store.grant(OWNER, world.root, "agent-1", (CaseOperation.INSPECT, CaseOperation.ACTION))
        run = world.act(AGENT, world.root, "run-1", RUN_START)
        self.host_session = world.act(AGENT, run.digest, "add-1", ADD).digest


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    return Tree(tmp_path / "store")


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def test_published_schemas_match_their_types() -> None:
    for name, schema in dataset_json_schemas().items():
        assert json.loads((WORKTREE / "schemas" / name).read_text()) == schema, name
        example = WORKTREE / "schemas/examples" / name.replace(".json", ".example.json")
        assert not list(Draft202012Validator(schema).iter_errors(json.loads(example.read_text()))), example


def test_every_exported_episode_validates_against_the_schema_and_replays_from_the_record(tree: Tree) -> None:
    bundle = export_episodes(tree.world.store, [tree.world.root], INPUTS)
    validator = Draft202012Validator(episode_json_schema())
    rows = [e.model_dump(mode="json") for e in bundle.episodes]

    assert {e.leaf for e in bundle.episodes} == {tree.answered, tree.corrected, tree.incomplete, tree.unsuccessful}
    assert [list(validator.iter_errors(row)) for row in rows] == [[]] * 4
    assert bundle == export_episodes(tree.world.store, [tree.world.root], INPUTS)
    for record in bundle.episodes:
        assert playback_states(record) == tuple(step.state_after for step in record.steps)
        assert all(d.status == "unreviewed" and d.decided_by is None for d in record.eligibility)
        assert all(r.training_use.right == "unknown" for r in record.rights)
    corrected = next(e for e in bundle.episodes if e.leaf == tree.corrected)
    assert corrected.provenance == "expert_reviewed_automated" and corrected.reference_exposed
    assert [c.kind for c in corrected.human_contributions] == ["phase_edit", "refine"]
    assert corrected.status.status == "submitted" and corrected.status.verdict == "NOT_RECORDED"
    assert bundle.manifest.excluded_by_reason == {"development_session": 1, "open": 1}
    assert [(e.leaf, e.reasons) for e in bundle.excluded] == [(tree.host_session, ("open", "development_session"))]


def test_demo_tools_remap_public_episode_ids(tree: Tree) -> None:
    rewrite = runpy.run_path(str(WORKTREE / "tools/demo/rewrite_store.py"))
    row = export_episodes(tree.world.store, [tree.world.root], INPUTS).episodes[0].model_dump(mode="json")
    sizes = {path.name: path.stat().st_size for path in (tree.world.store.root / "blobs").iterdir()}
    changed = rewrite["remap_json"](row, {row["root_digest"]: "a" * 64}, sizes)

    assert changed["record_id"] != row["record_id"]
    assert changed["record_id"] == exports._record_id({key: value for key, value in changed.items() if key != "record_id"})
    rewrite["check_integrity"](changed, sizes, "episodes.jsonl")
    changed["record_id"] = "0" * 64
    with pytest.raises(ValueError, match="record_id disagrees"):
        rewrite["check_integrity"](changed, sizes, "episodes.jsonl")


def test_an_imported_run_missing_replay_evidence_is_marked_incomplete_never_synthesized(tree: Tree) -> None:
    bundle = export_episodes(tree.world.store, [tree.world.root], INPUTS)
    incomplete = next(e for e in bundle.episodes if e.leaf == tree.incomplete)
    complete = next(e for e in bundle.episodes if e.leaf == tree.answered)
    outcome = incomplete.steps[-1].attempt.outcome

    assert [run.complete for run in incomplete.runs] == [False]
    assert outcome.result.completeness.missing == ("basis fit: output/run1/run1.lst",)
    assert outcome.result.basis_fit is None
    assert incomplete.steps[-1].state_after.hypothesis.fit is None
    assert incomplete.runs[0].consumed.cost_usd is None and incomplete.effort.cost_usd is None
    assert [run.complete for run in complete.runs] == [True]


# --- the dataset: snapshots, labels by rater kind, judge inputs ---------------------------------


def fixture_snapshot(store: WorkbenchStore, principal: Principal, origin: str, kind: str) -> AnalysisSnapshot:
    """Stands in for snapshot.build_snapshot (worker W3) over the fixture tree: the answered
    hypothesis and fit at the origin, with no checks."""
    state = store.view(principal, origin).state
    hypothesis = state.hypothesis
    if hypothesis is None or hypothesis.fit is None:
        raise LookupError("no fit to freeze")
    fit = hypothesis.fit
    weights = {p.candidate_id: p.weight_percent for p in fit.phases}
    return AnalysisSnapshot(
        kind=kind,
        origin=origin,
        root=state.root,
        case_id=CASE.case_id,
        pattern_sha256=CASE.pattern.sha256,
        fit_curve_sha256=fit.curve.sha256 if fit.curve else None,
        stage=fit.stage,
        rwp_percent=fit.rwp_percent,
        gof=fit.gof,
        quantification=fit.quantification,
        phases=tuple(
            SnapshotPhase(
                phase_id=p, formula=FORMULAS[p], space_group=None, weight_percent=weights.get(p), weight_percent_error=None
            )
            for p in hypothesis.phases
        ),
        residual_peaks=(ResidualPeak(peak_id="p10.02", two_theta_deg=10.02, height_percent=6.0, residual_counts=40.0),),
        checks=(),
        answered=True,
    )


def fixture_review(store: WorkbenchStore, principal: Principal, snap: AnalysisSnapshot, digest: str) -> dict:
    """Stands in for exports.blind_view (W3's blind payload) over the fixture tree: the brief,
    answer, fit curve and missing list of the expert's blind view."""
    rows = [line.split(",") for line in store._read_blob(snap.fit_curve_sha256).decode().splitlines()[1:]]
    curve = {name: [float(r[i]) for r in rows] for i, name in enumerate(("two_theta", "observed", "calculated", "difference"))}
    return {
        "brief": {"task": CASE.task, "precursors": [{"formula": "Li2CO3", "mass_g": None}]},
        "answer": {"phases": [p.phase_id for p in snap.phases], "claims": []},
        "fit": curve
        | {
            "stride": 1,
            "points": len(rows),
            "native_points": len(rows),
            "precision": "4 significant digits",
            "curve_sha256": snap.fit_curve_sha256,
            "phase_curves": {},
        },
        "residual_windows": [{"peak_id": p.peak_id, "two_theta_deg": [p.two_theta_deg]} for p in snap.residual_peaks],
        "phase_windows": [{"phase_id": p.phase_id, "center_deg": 10.02, "two_theta_deg": [10.0, 10.02]} for p in snap.phases],
        "phase_window_rule": "the fixture's rule",
        "missing": ["brief.precursors[].mass_g"],
        "steps": [{"title": "refine"}],
    }


JUDGE_ID = JudgeIdentity(
    model="claude-opus-5-5", prompt_sha256="7" * 64, input_sha256="8" * 64, run_id="judge-run-1", judge_is_actor_model=False
)
PERMIT = PermissionRequest(
    participation_and_storage=True,
    internal_evaluation_and_calibration=True,
    public_release=False,
    training=False,
    public_identity_disclosure=False,
)


class Labeled:
    def __init__(self, tree: Tree, tmp_path: Path) -> None:
        self.tree = tree
        self.store = tree.world.store
        self.labels = LabelStore(self.store)
        self.pseudonyms = RaterPseudonyms(self.store.root / "private")
        self.judge = Principal("judge-opus", PrincipalType.MODEL_JUDGE, frozenset())
        self.out = tmp_path / "dataset"
        for rater in (RATER, OWNER_RATER):
            self.store.grant(OWNER, tree.world.root, rater.principal_id, (CaseOperation.INSPECT,))

    def build(self, out: Path | None = None) -> exports.Dataset:
        return self.build_with(out=out)

    def build_with(self, out: Path | None = None, **options: object) -> exports.Dataset:
        dataset = build_dataset(
            self.store,
            self.labels,
            OWNER,
            [self.tree.world.root],
            INPUTS,
            "internal_evaluation",
            self.pseudonyms,
            **({"build_snapshot": fixture_snapshot, "review_payload": fixture_review} | options),
        )
        destination = out or self.out
        if destination.exists():
            shutil.rmtree(destination)
        write_dataset(dataset, destination)
        return dataset

    def label_both_kinds(self, digest: str) -> None:
        # A rater reads only its cohort before the reveal (CR-07): assign both raters first.
        raters = (OWNER_RATER.principal_id, RATER.principal_id)
        self.labels.assign(OWNER, self.tree.world.root, raters, (digest,))
        for principal in (OWNER_RATER, RATER):
            self.labels.record_permission(principal, PERMIT)
            self.labels.label(principal, LabelRequest(snapshot=digest, item="fit_supports_conclusion", answer="yes"))
        self.labels.judge_label(
            self.judge,
            Label(
                snapshot=digest,
                item="fit_supports_conclusion",
                phase_id=None,
                answer="no",
                rater=Rater(principal_id="judge-opus", kind="model_judge", judge=JUDGE_ID),
                exposure="blind",
                created_at_utc=NOW,
            ),
        )


    def record_every_kind(self, snapshot: AnalysisSnapshot) -> None:
        """One record of each kind, for the schema examples."""
        digest = exports.digest_of(snapshot)
        self.label_both_kinds(digest)
        self.labels.peak_call(OWNER_RATER, PeakCallRequest(snapshot=digest, peak_id="p10.02", cause="noise_or_profile"))
        self.labels.sign_off(OWNER_RATER, SignOffRequest(snapshot=digest, verdict="would_sign"))
        step = StepRef(attempt=snapshot.origin, call_index=0, step_kind="tool_call", label="refine CIF000001,CIF000002")
        self.labels.step_feedback(OWNER_RATER, StepFeedbackRequest(step=step, mark="good_move"))


@pytest.fixture
def labeled(tree: Tree, tmp_path: Path) -> Labeled:
    return Labeled(tree, tmp_path)


def test_model_judge_and_expert_labels_land_in_separate_files(labeled: Labeled) -> None:
    first = labeled.build()
    (digest,) = {s.origin: exports.digest_of(s) for s in first.snapshots if s.origin == labeled.tree.answered}.values()
    labeled.label_both_kinds(digest)
    dataset = labeled.build()
    out = labeled.out

    assert validate_dataset(out, dataset_json_schemas()) == []
    assert sorted(r["rater"]["kind"] for r in lines(out / "labels.expert.jsonl")) == ["expert", "owner"]
    assert [r["rater"]["kind"] for r in lines(out / "labels.model_judge.jsonl")] == ["model_judge"]
    counts = {c.name: (c.count, c.denominator) for c in dataset.manifest.counts}
    assert counts["labels_expert"] == (2, 3) and counts["labels_model_judge"] == (1, 3)
    assert counts["snapshots_with_model_judge_labels"] == (1, 1)

    judge_line = (out / "labels.model_judge.jsonl").read_bytes()
    (out / "labels.expert.jsonl").write_bytes((out / "labels.expert.jsonl").read_bytes() + judge_line)
    assert any("model_judge does not belong here" in p for p in validate_dataset(out, dataset_json_schemas()))


def test_the_export_carries_no_custody_field_or_run_identity(labeled: Labeled) -> None:
    dataset = labeled.build()
    out = labeled.out
    schemas = dataset_json_schemas()
    canary = "AXA-CANARY-0123456789abcdef"

    assert validate_dataset(out, schemas, canaries=[canary]) == []
    assert [s.origin for s in dataset.snapshots] == [labeled.tree.answered]
    snapshot_exclusions = {e.ref for e in dataset.manifest.exclusions if e.kind == "snapshot"}
    assert snapshot_exclusions == {labeled.tree.incomplete}  # answered, but no fit to freeze
    # KTD10: a run that ended without an answer keeps its episode and gets no snapshot.
    assert labeled.tree.unsuccessful in {e.leaf for e in dataset.episodes}
    (judge_input,) = lines(out / "judge_inputs.jsonl")
    assert set(judge_input) == set(exports.JudgeInput.model_fields)
    assert judge_input["reported_phase_ids"] == ["CIF000001", "CIF000002"]
    assert judge_input["fit"]["curve_sha256"] == hashlib.sha256(CURVE).hexdigest()
    assert [w["peak_id"] for w in judge_input["residual_windows"]] == ["p10.02"]
    assert [w["phase_id"] for w in judge_input["phase_windows"]] == ["CIF000001", "CIF000002"]
    assert judge_input["phase_window_rule"] == "the fixture's rule"
    assert judge_input["fit"]["observed"] == [120, 131] and judge_input["fit"]["difference"] == [2, 1]

    snapshot_line = json.loads((out / "snapshots.jsonl").read_bytes())
    leaked = snapshot_line | {"claims": [canary]}
    (out / "snapshots.jsonl").write_bytes(json.dumps(leaked).encode() + b"\n")
    judge_input["brief"]["model"] = "qwen36-27b"
    judge_input["reference_answer"] = {"phases": ["Li2CO3"]}
    (out / "judge_inputs.jsonl").write_bytes(json.dumps(judge_input).encode() + b"\n")
    problems = validate_dataset(out, schemas, canaries=[canary])

    assert any("snapshots.jsonl: carries a custody canary" in p for p in problems)
    assert any("run identity field ['model']" in p for p in problems)
    assert any("custody field ['reference_answer']" in p for p in problems)


def test_exported_episodes_carry_no_expert_only_field(labeled: Labeled, tmp_path: Path, monkeypatch) -> None:
    world = labeled.tree.world
    world.act(OWNER, labeled.tree.corrected, "private-reason", ADD, rationale="Expert interpretation sentinel")
    author = Principal(OWNER.principal_id, PrincipalType.USER, frozenset({"rater"}))
    labeled.labels.record_permission(author, PERMIT.model_copy(update={"public_release": True}))
    monkeypatch.setattr(cli, "open_store", lambda *args: world.store)
    monkeypatch.setattr(cli, "OWNER", OWNER)
    monkeypatch.setattr(cli.snapshot, "build_snapshot", fixture_snapshot)
    monkeypatch.setattr(exports, "blind_view", fixture_review)
    out = tmp_path / "cli-export"

    assert cli.main(["export", "--store", str(world.store.root), "--out", str(out)]) == 0

    rows = lines(out / "episodes.jsonl")
    assert rows
    assert "Expert interpretation sentinel" not in (out / "episodes.jsonl").read_text()
    for row in rows:
        assert row["schema_version"] == "phaseeval.episode.v2"
        assert "ingest" not in row["root"]["case"]
        assert all("rationale" not in step["attempt"] for step in row["steps"])
        assert all("idempotency_key" not in step["attempt"] for step in row["steps"])
        assert all("source_library_id" not in candidate for candidate in row["root"]["case"]["candidates"])
        assert "original_name" not in set(exports._keys(row))
    assert validate_dataset(out, dataset_json_schemas()) == []


@pytest.mark.parametrize("field,value,reason", [
    (("task",), "", "at least 1 character"),
    (("task",), "x" * 8001, "at most 8000 characters"),
    (("candidates",), [], "at least 1 item"),
    (("candidates", 0, "candidate_id"), "../candidate", "String should match pattern"),
    (("candidates", 0, "elements"), ["invalid"], "String should match pattern"),
    (("candidates", 1, "candidate_id"), "CIF000001", "candidate ids are unique"),
    (("pattern", "role"), "custody_withheld", "pattern artifact has role observed_pattern"),
    (("candidates", 0, "cif", "role"), "custody_withheld", "every candidate CIF has role candidate_cif"),
    (("source", "collection"), "xrdbench", "protected evaluation collections"),
    (("source", "collection"), "terminal_bench_science", "protected evaluation collections"),
    (("source", "collection"), "dara_zenodo_17410051", "benchmark cases come from dev systems"),
    (("source", "collection"), "precursor_genome_zenodo_21285546", "benchmark cases come from dev systems"),
], ids=["empty-task", "long-task", "empty-candidates", "candidate-id", "element", "duplicate-candidates",
        "pattern-role", "cif-role", "xrdbench", "terminal-bench", "dara-split", "precursor-split"])
def test_public_episode_retains_case_validation(labeled: Labeled, field, value, reason) -> None:
    labeled.build()
    path = labeled.out / "episodes.jsonl"
    rows = lines(path)
    case = rows[0]["root"]["case"]
    for key in field[:-1]:
        case = case[key]
    case[field[-1]] = value
    for row in rows:
        body = {key: value for key, value in row.items() if key != "record_id"}
        row["record_id"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    data = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path.write_bytes(data)
    manifest_path = labeled.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    next(entry for entry in manifest["files"] if entry["path"] == path.name)["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    problems = validate_dataset(labeled.out, dataset_json_schemas())

    assert any("episodes.jsonl:1: model:" in problem for problem in problems), (reason, problems)


def test_a_withdrawn_raters_labels_vanish_from_the_next_export_and_ids_stay_pseudonymous(labeled: Labeled, tmp_path: Path) -> None:
    first = labeled.build()
    digest = exports.digest_of(first.snapshots[0])
    labeled.label_both_kinds(digest)
    before = labeled.build(tmp_path / "before")
    labeled.labels.record_permission(RATER, PERMIT.model_copy(update={"participation_and_storage": False}))
    after = labeled.build(tmp_path / "after")
    identity_map = labeled.pseudonyms.write_map()
    ada = labeled.pseudonyms("rater-ada")

    assert ada in {r.rater.principal_id for r in before.labels_expert}
    assert ada not in {r.rater.principal_id for r in after.labels_expert}
    assert [r.rater.kind for r in after.labels_expert] == ["owner"]
    for directory in (tmp_path / "before", tmp_path / "after"):
        for path in directory.iterdir():
            data = path.read_bytes()
            assert b"rater-ada" not in data and b"owner-rater" not in data, path
            assert OWNER.principal_id.encode() not in data, path
    assert identity_map.parent == labeled.store.root / "private"
    assert json.loads(identity_map.read_text())[ada] == "rater-ada"
    assert identity_map.stat().st_mode & 0o077 == 0


def test_a_snapshot_frozen_under_an_older_type_is_exported_byte_for_byte(labeled: Labeled) -> None:
    """A later AnalysisSnapshot type may add defaulted fields; re-serializing an older frozen
    snapshot through it would insert them and change the evidence its digest names."""
    snap = fixture_snapshot(labeled.store, OWNER, labeled.tree.answered, "agent_answer")
    older = json.loads(exports.canonical_bytes(snap))
    del older["chemistry"]  # a field the current type defaults, absent when this snapshot froze
    data = json.dumps(older, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    (labeled.store.root / "snapshots").mkdir(mode=0o700, exist_ok=True)
    (labeled.store.root / "snapshots" / f"{digest}.json").write_bytes(data)

    dataset = labeled.build()

    assert (labeled.out / "snapshots.jsonl").read_bytes() == data
    (judge_input,) = lines(labeled.out / "judge_inputs.jsonl")
    assert judge_input["snapshot_digest"] == digest and judge_input["snapshot"] == older
    assert [e.snapshot_digest for e in dataset.judge_manifest.entries] == [digest]
    assert validate_dataset(labeled.out, dataset_json_schemas()) == []


def test_a_frozen_judge_input_never_changes_and_a_later_snapshot_is_not_judged(labeled: Labeled) -> None:
    assert labeled.build().judge_manifest.frozen is False  # provisional until the judges launch
    first = labeled.build_with(freeze_judge_manifest=True)
    assert first.judge_manifest.frozen is True
    (frozen,) = first.judge_manifest.entries
    later_answer = import_run(
        labeled.tree.world, "task-dev900__d", end=ANSWERED, completeness={"kind": "complete"}, basis_fit=BASIS_FIT
    )
    correction = fixture_snapshot(labeled.store, OWNER, labeled.tree.corrected, "expert_correction")
    correction_digest = labeled.labels.put_snapshot(correction)
    second = labeled.build_with(judge_manifest=first.judge_manifest)
    (later,) = [s for s in second.snapshots if s.origin == later_answer]

    assert [(e.snapshot_digest, e.judged) for e in second.judge_manifest.entries] == [
        (frozen.snapshot_digest, True),
        (exports.digest_of(later), False),
    ]
    assert second.judge_manifest.entries[0] == frozen
    # CR-08: an expert correction is exported for raters but never reaches a judge.
    assert correction_digest in {e.digest for e in second.manifest.snapshot_index}
    assert correction_digest not in {j.snapshot_digest for j in second.judge_inputs}
    assert validate_dataset(labeled.out, dataset_json_schemas()) == []

    def drifted(store: WorkbenchStore, principal: Principal, snap: AnalysisSnapshot, digest: str) -> dict:
        return fixture_review(store, principal, snap, digest) | {"missing": ["regenerated after the freeze"]}

    with pytest.raises(ValueError, match="changed after the judge manifest froze"):
        labeled.build_with(judge_manifest=first.judge_manifest, review_payload=drifted)


def test_the_private_export_carries_qualification_expectations_and_reports_and_the_public_neither(
    labeled: Labeled, tmp_path: Path
) -> None:
    labeled.build()
    qualifier = Qualifier(labeled.store, labeled.labels, NoReference())
    policy = QualificationPolicy(
        version="lab-policy-test",
        justified_by="fixture",
        rater_kinds=("owner",),
        require_blind=False,
        min_raters=1,
        phase_identity="formula_and_space_group",
        amount_abs_wt_pct=None,
        expectation_rule="lab-expectation-v2",
    )
    set_digest = qualifier.freeze(OWNER, lab="fixture-lab", name="dev900", roots=(labeled.tree.world.root,), policy=policy, split_seed=None)
    report = qualifier.run(OWNER, set_digest, Producer(model_id="qwen36-27b", harness="harbor-local-mcp", harness_version="2"))
    private = labeled.build_with(qualifier=qualifier)
    public = build_dataset(
        labeled.store, labeled.labels, OWNER, [labeled.tree.world.root], INPUTS, "public_release", labeled.pseudonyms,
        build_snapshot=fixture_snapshot, review_payload=fixture_review, scope="public", qualifier=qualifier,
    )
    write_dataset(public, tmp_path / "public")
    schemas = dataset_json_schemas()

    assert validate_dataset(labeled.out, schemas) == []
    assert validate_dataset(tmp_path / "public", schemas) == []
    (entry,) = private.manifest.qualification_sets
    assert (entry.set_digest, entry.roots, entry.expectations_sha256) == (set_digest, 1, report.expectations_sha256)
    assert [r.model_dump(mode="json") for r in private.reports] == [report.model_dump(mode="json")]
    assert len(lines(labeled.out / "expectations.jsonl")) == 1 and len(lines(labeled.out / "reports.jsonl")) == 1
    assert not (tmp_path / "public/expectations.jsonl").exists() and not (tmp_path / "public/reports.jsonl").exists()
    assert public.manifest.qualification_sets == ()

    (tmp_path / "public/expectations.jsonl").write_bytes((labeled.out / "expectations.jsonl").read_bytes())
    assert "expectations.jsonl: a public bundle carries no grader file" in validate_dataset(tmp_path / "public", schemas)


# --- the real producers: importer, W3 snapshots and blind view, W2 label store -------------------

CUSTODY = Path(os.environ.get("AXA_WORKBENCH_CUSTODY", WORKTREE / "data/custody/workbench"))
QWEN_V2 = CUSTODY / "runs/jobs/u6-dev006-qwen-006/task-dev006__5SKiLMp"


@pytest.mark.skipif(
    not (CUSTODY / "DEV006/custody.json").is_file() or not QWEN_V2.is_dir(),
    reason=f"custody records or U6 runs absent under {CUSTODY}; set AXA_WORKBENCH_CUSTODY",
)
def test_a_real_import_builds_a_valid_dataset_through_the_real_snapshot_and_blind_view(tmp_path: Path) -> None:
    owner = Principal("owner", PrincipalType.USER, frozenset({"owner"}))
    importer_service = Principal("importer", PrincipalType.SERVICE, frozenset({"importer"}))
    store = WorkbenchStore(
        tmp_path / "store", default_registry().run, public_seams(), lambda: NOW,
        import_root=CUSTODY / "runs", exposure=ExposureLedger(tmp_path / "store"),
    )
    case, blobs = importer.read_case_package(
        WORKTREE / "data/workbench/cases/DEV006", WORKTREE / "data/workbench/case_labels.json", custody=CUSTODY / "DEV006/custody.json"
    )
    root = store.load_case(owner, case, blobs)
    run, result, run_blobs = importer.read_harbor_run(QWEN_V2, case)
    origin = store.import_run(importer_service, root, run, result, run_blobs)
    inputs = INPUTS.model_copy(update={"rights": policy.rights_registry_v0((case,))})
    labels = LabelStore(store)
    pseudonyms = RaterPseudonyms(store.root / "private")
    canaries = [json.loads((CUSTODY / c / "custody.json").read_text())["canary"] for c in ("DEV006",)]

    datasets = {}
    for scope, use in (("private", "internal_evaluation"), ("public", "public_release")):
        datasets[scope] = build_dataset(
            store,
            labels,
            owner,
            [root],
            inputs,
            use,
            pseudonyms,
            build_snapshot=snapshot.build_snapshot,
            review_payload=exports.blind_view,
            scope=scope,
            judge_manifest=datasets["private"].judge_manifest if scope == "public" else None,
        )
        write_dataset(datasets[scope], tmp_path / scope)
        assert validate_dataset(tmp_path / scope, dataset_json_schemas(), canaries) == [], scope

    (judge_input,) = lines(tmp_path / "private/judge_inputs.jsonl")
    snap = judge_input["snapshot"]
    assert snap["origin"] == origin and snap["kind"] == "agent_answer" and snap["answered"]
    assert {p["phase_id"] for p in snap["phases"]} == {"CIF001783", "CIF012075", "COD2310703"}
    assert snap["chemistry"] is not None
    assert len(judge_input["fit"]["observed"]) == judge_input["fit"]["points"] <= 1200
    assert set(judge_input["fit"]["phase_curves"]) == {p["phase_id"] for p in snap["phases"]}
    entries = datasets["private"].judge_manifest.entries
    assert [(e.snapshot_digest, e.judged, e.actor_model) for e in entries] == [(judge_input["snapshot_digest"], True, "qwen36-27b")]
    assert judge_input["reported_phase_ids"] == [p["phase_id"] for p in snap["phases"]]
    assert judge_input["fit"]["native_points"] > judge_input["fit"]["points"]
    windows = {w["peak_id"]: w for w in judge_input["residual_windows"]}
    assert set(windows) == {p["peak_id"] for p in snap["residual_peaks"]}
    for window in windows.values():  # native resolution with every fitted phase's contribution
        assert set(window["phase_curves"]) == {p["phase_id"] for p in snap["phases"]}
    # Science (AXA-43zwk.10 08:42): each reported phase's own calculated peaks, shared with the page
    by_phase = {}
    for window in judge_input["phase_windows"]:
        by_phase.setdefault(window["phase_id"], []).append(window)
        assert set(window["phase_curves"]) == {p["phase_id"] for p in snap["phases"]}
    assert set(by_phase) == {p["phase_id"] for p in snap["phases"]}
    assert all(1 <= len(ws) <= snapshot.PHASE_WINDOWS for ws in by_phase.values())
    assert judge_input["phase_window_rule"] == snapshot.PHASE_WINDOW_RULE
    blind = exports.blind_view(store, owner, exports.AnalysisSnapshot.model_validate(snap), judge_input["snapshot_digest"])
    assert {key: judge_input[key] for key in exports.JUDGE_VIEW_KEYS} == json.loads(
        json.dumps({key: blind[key] for key in exports.JUDGE_VIEW_KEYS})
    )  # parity: the judge reads exactly the expert's blind view
    submitted = judge_input["answer"]["submitted"]
    assert submitted is not None and submitted == snap["reported_answer"]  # one answer projection
    assert "fit_statistics" in submitted and "fit_artifacts" not in submitted
    assert [e.judged for e in datasets["public"].manifest.snapshot_index] == [True]
    assert len(datasets["public"].episodes) == 1  # u6-dev006-qwen-006 is an origin-Q6 release job
