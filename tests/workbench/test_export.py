"""U6 episode export and the eval dataset: schema, exclusions, completeness, label separation,
custody and withdrawal (R11, R18, R19, R33; overnight N4, N8)."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from tests.support.store import ADD, BLOBS, CASE, OWNER, World

from phaseeval.api.auth import Principal, PrincipalType
from phaseeval.workbench import cli, exports, policy, snapshot
from phaseeval.workbench.exports import (
    RaterPseudonyms,
    build_dataset,
    dataset_json_schemas,
    episode_json_schema,
    export_episodes,
    playback_states,
    validate_dataset,
    write_dataset,
)
from phaseeval.workbench.labels import LabelStore
from phaseeval.workbench.model import CaseManifest, ChildAttempt
from phaseeval.workbench.qualification import Producer, QualificationPolicy, Qualifier
from phaseeval.workbench.seams import NoReference
from phaseeval.workbench.snapshot import AnalysisSnapshot
from phaseeval.workbench.store import WorkbenchStore

from tests.support.runs import (
    ANSWERED,
    BASIS_FIT,
    CURVE,
    INPUTS,
    Labeled,
    PERMIT,
    RATER,
    Tree,
    WORKTREE,
    import_run,
    lines,
)
from tests.support.runs import _ref











@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    return Tree(tmp_path / "store")




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
    assert counts["snapshots_with_model_judge_labels"] == (1, 2)

    judge_line = (out / "labels.model_judge.jsonl").read_bytes()
    (out / "labels.expert.jsonl").write_bytes((out / "labels.expert.jsonl").read_bytes() + judge_line)
    assert any("model_judge does not belong here" in p for p in validate_dataset(out, dataset_json_schemas()))


def test_the_export_carries_no_custody_field_or_run_identity(labeled: Labeled) -> None:
    dataset = labeled.build()
    assert {c.code for c in next(s for s in dataset.snapshots if s.origin == labeled.tree.answered).checks} == {"K1", "K2", "K3", "K5"}
    out = labeled.out
    schemas = dataset_json_schemas()
    canary = "AXA-CANARY-0123456789abcdef"

    assert validate_dataset(out, schemas, canaries=[canary]) == []
    assert {s.origin for s in dataset.snapshots} == {labeled.tree.answered, labeled.tree.incomplete}
    snapshot_exclusions = {e.ref for e in dataset.manifest.exclusions if e.kind == "snapshot"}
    assert snapshot_exclusions == set()
    # KTD10: a run that ended without an answer keeps its episode and gets no snapshot.
    assert labeled.tree.unsuccessful in {e.leaf for e in dataset.episodes}
    (judge_input,) = [j for j in lines(out / "judge_inputs.jsonl") if j["snapshot"]["origin"] == labeled.tree.answered]
    assert set(judge_input) == set(exports.JudgeInput.model_fields)
    assert judge_input["reported_phase_ids"] == ["CIF000001", "CIF000002"]
    assert judge_input["fit"]["curve_sha256"] == hashlib.sha256(CURVE).hexdigest()
    assert [w["peak_id"] for w in judge_input["residual_windows"]] == ["p23.95"]
    assert [w["phase_id"] for w in judge_input["phase_windows"]] == ["CIF000001", "CIF000002"]
    assert judge_input["phase_window_rule"] == snapshot.PHASE_WINDOW_RULE
    assert judge_input["fit"]["native_points"] == 1001
    assert max(judge_input["fit"]["difference"]) == 3180

    snapshot_line = next(row for row in lines(out / "snapshots.jsonl") if row["origin"] == labeled.tree.answered)
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
    out = tmp_path / "cli-export"

    assert cli.main(["export", "--store", str(world.store.root), "--out", str(out)]) == 0

    rows = lines(out / "episodes.jsonl")
    assert rows
    for row in rows:
        encoded = json.dumps(row)
        assert '"channel": "custody_record"' not in encoded
        assert '"channel": "operator_note"' not in encoded
        assert row["root"]["case"]["material"]["precursors"][0]["amount"] is None
        assert all(step["state_after"]["material"]["precursors"][0]["amount"] is None for step in row["steps"])
    assert "Expert interpretation sentinel" not in (out / "episodes.jsonl").read_text()
    for row in rows:
        assert row["schema_version"] == "phaseeval.episode.v2"
        assert "ingest" not in row["root"]["case"]
        assert all("rationale" not in step["attempt"] for step in row["steps"])
        assert all("idempotency_key" not in step["attempt"] for step in row["steps"])
        assert all("source_library_id" not in candidate for candidate in row["root"]["case"]["candidates"])
        assert "original_name" not in set(exports._keys(row))
    assert validate_dataset(out, dataset_json_schemas()) == []


@pytest.mark.parametrize("location", ["artifacts", "basis_curve", "receipts", "final_curve", "stage_curve"])
def test_export_omits_withheld_artifacts_at_every_outcome_depth(tree: Tree, monkeypatch, location) -> None:
    walked = tree.world.store.walk(tree.world.root)
    target = tree.answered if location in {"basis_curve", "receipts"} else tree.corrected
    withheld = _ref(b"withheld evidence", "text/csv", "custody_withheld")
    changed = []
    for digest, attempt in walked:
        if digest == target:
            data = attempt.model_dump(mode="json")
            outcome = data["outcome"]
            if location == "artifacts":
                outcome["artifacts"].append(withheld)
            elif location == "receipts":
                outcome["origin"]["receipts"].append(withheld)
            elif location == "stage_curve":
                outcome["result"]["stages"] = [outcome["result"]["final"] | {
                    "curve": withheld, "rexp_percent": None, "converged": None,
                    "iterations": 1, "measured_points": None, "peaks": None, "parameters": None,
                }]
                outcome["result"]["stages"][0].pop("quantification")
            else:
                outcome["result"]["basis_fit" if location == "basis_curve" else "final"]["curve"] = withheld
            attempt = ChildAttempt.model_validate(data)
        changed.append((digest, attempt))
    monkeypatch.setattr(tree.world.store, "walk", lambda root: changed)

    bundle = export_episodes(tree.world.store, [tree.world.root], INPUTS)

    for episode in bundle.episodes:
        row = episode.model_dump(mode="json")
        assert "custody_withheld" not in json.dumps(row)
        assert withheld["sha256"] not in json.dumps(row)
        assert not list(Draft202012Validator(episode_json_schema()).iter_errors(row))
        assert playback_states(episode) == tuple(step.state_after for step in episode.steps)
    assert any(step.attempt.outcome.origin.receipts for episode in bundle.episodes for step in episode.steps
               if step.digest == tree.answered)


@pytest.mark.parametrize("channel", ["custody_record", "operator_note"])
def test_export_filters_every_nested_context_value(tmp_path: Path, channel) -> None:
    world = World(tmp_path / "store")
    case = CASE.model_dump(mode="json")
    material = case["material"]
    source = material["precursors"][0]["amount"]["source"] | {"channel": channel}
    precursor = material["precursors"][0]
    for name, value in {"amount": {"value": 1.5, "unit": "g"}, "target_amount": {"value": 2, "unit": "g"},
                        "name": "hidden name", "cas_number": "hidden cas", "purity": 0.99}.items():
        precursor[name] = {"value": value, "source": source}
    for name in ("declared_target", "declared_target_stoichiometry", "specimen"):
        material[name] = {"value": "hidden material", "source": source}
    processing = {"order": 1, "operation": {"value": "heat", "source": precursor["formula"]["source"]}}
    for name, value in {"equipment": "hidden equipment", "furnace": "hidden furnace", "temperature_c": 600,
                        "duration_min": 10, "atmosphere": "hidden atmosphere", "heating_rate_c_per_min": 5,
                        "temperature_profile": {"time_min": [0, 10], "temperature_c": [20, 600]},
                        "cooling": "air_quench", "cooling_rate_c_per_min": 4,
                        "met_target_mass": True, "mass_mg": 1500}.items():
        processing[name] = {"value": value, "source": source}
    material["processing"] = [processing]
    world.root = world.store.load_case(OWNER, CaseManifest.model_validate(case), BLOBS)
    import_run(world, "private-context", end=ANSWERED, completeness={"kind": "complete"}, basis_fit=BASIS_FIT)

    bundle = export_episodes(world.store, [world.root], INPUTS)

    assert len(bundle.episodes) == 1
    episode = bundle.episodes[0]
    assert f'"channel": "{channel}"' not in episode.model_dump_json(indent=2)
    assert episode.root.case.material.precursors[0].formula == CASE.material.precursors[0].formula
    assert episode.root.case.measurement == CASE.measurement
    assert playback_states(episode) == tuple(step.state_after for step in episode.steps)


@pytest.mark.parametrize("context,field", [("measurement", "instrument"), ("measurement", "radiation"),
                                          ("measurement", "profile"), ("material", "elements")])
def test_export_refuses_required_expert_only_context(tmp_path: Path, context, field) -> None:
    world = World(tmp_path / "store")
    case = CASE.model_dump(mode="json")
    case[context][field]["source"]["channel"] = "custody_record"
    world.root = world.store.load_case(OWNER, CaseManifest.model_validate(case), BLOBS)
    import_run(world, "private-required-context", end=ANSWERED, completeness={"kind": "complete"}, basis_fit=BASIS_FIT)

    with pytest.raises(ValueError, match=field):
        export_episodes(world.store, [world.root], INPUTS)


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
    snap = snapshot.build_snapshot(labeled.store, OWNER, labeled.tree.answered, "agent_answer")
    older = json.loads(exports.canonical_bytes(snap))
    del older["chemistry"]  # a field the current type defaults, absent when this snapshot froze
    data = json.dumps(older, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    (labeled.store.root / "snapshots").mkdir(mode=0o700, exist_ok=True)
    (labeled.store.root / "snapshots" / f"{digest}.json").write_bytes(data)

    dataset = labeled.build()

    assert data in (labeled.out / "snapshots.jsonl").read_bytes().splitlines(keepends=True)
    (judge_input,) = [j for j in lines(labeled.out / "judge_inputs.jsonl") if j["snapshot_digest"] == digest]
    assert judge_input["snapshot_digest"] == digest and judge_input["snapshot"] == older
    assert digest in [e.snapshot_digest for e in dataset.judge_manifest.entries]
    assert validate_dataset(labeled.out, dataset_json_schemas()) == []


def test_a_frozen_judge_input_never_changes_and_a_later_snapshot_is_not_judged(labeled: Labeled) -> None:
    assert labeled.build().judge_manifest.frozen is False  # provisional until the judges launch
    first = labeled.build_with(freeze_judge_manifest=True)
    assert first.judge_manifest.frozen is True
    (frozen,) = [e for e in first.judge_manifest.entries if e.snapshot_digest == exports.digest_of(next(s for s in first.snapshots if s.origin == labeled.tree.answered))]
    later_answer = import_run(
        labeled.tree.world, "task-dev900__d", end=ANSWERED, completeness={"kind": "complete"}, basis_fit=BASIS_FIT
    )
    correction = snapshot.build_snapshot(labeled.store, OWNER, labeled.tree.corrected, "expert_correction")
    correction_digest = labeled.labels.put_snapshot(correction)
    second = labeled.build_with(judge_manifest=first.judge_manifest)
    (later,) = [s for s in second.snapshots if s.origin == later_answer]

    assert {e.snapshot_digest: e.judged for e in second.judge_manifest.entries} == {
        **{e.snapshot_digest: True for e in first.judge_manifest.entries},
        exports.digest_of(later): False,
    }
    assert frozen in second.judge_manifest.entries
    # CR-08: an expert correction is exported for raters but never reaches a judge.
    assert correction_digest in {e.digest for e in second.manifest.snapshot_index}
    assert correction_digest not in {j.snapshot_digest for j in second.judge_inputs}
    assert validate_dataset(labeled.out, dataset_json_schemas()) == []

    def drifted(store: WorkbenchStore, principal: Principal, snap: AnalysisSnapshot, digest: str) -> dict:
        return exports.blind_view(store, principal, snap, digest) | {"missing": ["regenerated after the freeze"]}

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
        build_snapshot=snapshot.build_snapshot, review_payload=exports.blind_view, scope="public", qualifier=qualifier,
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

def test_a_real_import_builds_a_valid_dataset_through_the_real_snapshot_and_blind_view(tmp_path, demo_store):
    owner = cli.OWNER
    store, (origin,) = demo_store.import_into(tmp_path / 'imported', [demo_store.origin()])
    root = store.roots(owner)[0]
    case = store.walk(root)[0][1].case
    inputs = INPUTS.model_copy(update={"rights": policy.rights_registry_v0((case,))})
    labels = LabelStore(store)
    pseudonyms = RaterPseudonyms(store.root / "private")
    canaries = []

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
