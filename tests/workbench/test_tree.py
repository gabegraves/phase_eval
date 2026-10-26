"""U16 attempt tree: the store's act path, idempotency, projections and crash convergence."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from phaseeval.api.auth import CaseOperation
from phaseeval.workbench.model import (
    BlobDigest,
    CaseManifest,
    Completed,
    Halt,
    ImportedRunResult,
    ImportRun,
    InfraFailed,
    InfraFailure,
    MalformedToolUse,
    Refused,
    Unparsed,
    state_sha256,
)
from phaseeval.workbench.review import ActorAttemptView, RevealedAttemptView
from phaseeval.workbench.store import (
    ActionRefused,
    ActionRequest,
    AttemptAccessError,
    IdempotencyConflict,
    WorkbenchStore,
)

from tests.support.store import ADD, AGENT, BLOBS, CASE, CASE_JSON, IMPORTER, OWNER, REFINE, RUN_START, World


def test_one_case_loads_to_one_root_digest_in_any_store_at_any_time(tmp_path: Path) -> None:
    first = World(tmp_path / "a")
    again = first.store.load_case(OWNER, CaseManifest.model_validate(CASE_JSON), dict(BLOBS))
    elsewhere = World(tmp_path / "b", start=datetime(2027, 1, 1, tzinfo=UTC))
    changed = CaseManifest.model_validate(CASE_JSON | {"task": "Quantify the phases."})

    assert again == first.root
    assert elsewhere.root == first.root
    assert first.store.load_case(OWNER, changed, BLOBS) != first.root
    assert first.tree() == [first.root]
    assert [a.kind for a in first.store.annotations(first.root)] == ["load_event"]


def test_one_key_posted_twice_is_one_attempt_and_one_execution(world: World) -> None:
    added = world.act(OWNER, world.root, "add-1", ADD)
    first = world.act(OWNER, added.digest, "refit-1", REFINE)
    retried = world.act(OWNER, added.digest, "refit-1", REFINE, rationale="network retry")

    assert isinstance(first, RevealedAttemptView) and isinstance(added, RevealedAttemptView)
    assert retried.digest == first.digest
    assert world.refine.calls == [("CIF000001", "CIF000002")]
    assert world.tree() == [world.root, added.digest, first.digest]
    assert first.state.hypothesis is not None
    assert first.state.hypothesis.fit_basis == first.digest
    assert first.state.hypothesis.phases == ("CIF000001", "CIF000002")
    outcome = first.attempt.outcome
    assert isinstance(outcome, Completed)
    assert outcome.parent_state_sha256 == state_sha256(added.state)
    assert outcome.custody_scan == "not_configured"


def test_one_key_with_different_arguments_is_a_conflict(world: World) -> None:
    added = world.act(OWNER, world.root, "add-1", ADD)
    world.act(OWNER, added.digest, "refit-1", REFINE)

    with pytest.raises(IdempotencyConflict):
        world.act(OWNER, added.digest, "refit-1", {"verb": "refine", "recipe": "pawley_only"})
    assert len(world.refine.calls) == 1


def test_agent_and_owner_posting_one_action_on_one_parent_get_distinct_attempts(world: World) -> None:
    world.store.grant(OWNER, world.root, "agent-1", (CaseOperation.INSPECT, CaseOperation.ACTION))
    run = world.act(AGENT, world.root, "run-1", RUN_START)
    by_agent = world.act(AGENT, run.digest, "edit-1", ADD)
    by_owner = world.act(OWNER, run.digest, "edit-1", ADD)

    assert by_agent.digest != by_owner.digest
    actors = {digest: attempt.actor.kind for digest, attempt in world.store.walk(world.root)[1:]}
    assert actors == {run.digest: "agent", by_agent.digest: "agent", by_owner.digest: "human"}
    revealed = world.store.view(OWNER, run.digest)
    assert isinstance(revealed, RevealedAttemptView)
    assert revealed.children == tuple(sorted((by_agent.digest, by_owner.digest)))


def test_an_actor_field_in_the_request_is_rejected_and_nothing_is_written(world: World) -> None:
    before = world.files()
    impostor = {"kind": "human", "principal_id": "someone-else", "reference_exposure": "unexposed"}

    with pytest.raises(ValidationError) as body:
        ActionRequest.model_validate({"idempotency_key": "k-1", "action": ADD, "actor": impostor})
    with pytest.raises(ActionRefused, match="add_phases.actor: Extra inputs are not permitted"):
        world.act(OWNER, world.root, "k-1", ADD | {"actor": impostor})

    assert [(e["type"], e["loc"]) for e in body.value.errors()] == [("extra_forbidden", ("actor",))]
    assert world.files() == before


def test_an_agent_gets_a_rejected_attempt_for_a_malformed_call_and_a_human_a_refusal(world: World) -> None:
    world.store.grant(OWNER, world.root, "agent-1", (CaseOperation.INSPECT, CaseOperation.ACTION))
    run = world.act(AGENT, world.root, "run-1", RUN_START)
    malformed = {"verb": "refine", "recipe": "fastest"}

    rejected = world.act(AGENT, run.digest, "bad-1", malformed)
    with pytest.raises(ActionRefused):
        world.act(OWNER, world.root, "bad-1", malformed)

    assert isinstance(rejected, ActorAttemptView)
    assert rejected.action == Unparsed(raw=malformed)
    assert rejected.outcome.status == "rejected"
    assert rejected.outcome.message == "refine.recipe: Input should be 'staged_rietveld' or 'pawley_only'"
    assert world.tree() == [world.root, run.digest, rejected.digest]


def test_a_continuation_agent_sees_the_expert_correction_as_sourced_context_and_nothing_else(
    world: World,
) -> None:
    rationale = "The operator note says Li2O; the recipe and the pattern say Li2CO3."
    correction = world.act(
        OWNER,
        world.root,
        "fix-target",
        {"verb": "correct_material", "patch": {"declared_target": "Li2CO3"}},
        rationale=rationale,
    )
    world.store.grant(OWNER, world.root, "agent-1", (CaseOperation.INSPECT, CaseOperation.ACTION))
    run = world.act(AGENT, correction.digest, "run-1", RUN_START)

    with pytest.raises(AttemptAccessError):
        world.store.view(AGENT, correction.digest)
    with pytest.raises(AttemptAccessError):
        world.store.path(AGENT, correction.digest)
    with pytest.raises(AttemptAccessError):
        world.act(AGENT, correction.digest, "probe-1", ADD)
    path = world.store.path(AGENT, run.digest)

    material = path.case.material  # type: ignore[union-attr]
    assert material.declared_target is not None
    assert material.declared_target.model_dump(mode="json") == {
        "value": "Li2CO3",
        "source": {"kind": "correction", "attempt": correction.digest, "principal_id": "owner-1"},
    }
    assert material.precursors[0].amount is None
    assert [step.digest for step in path.steps] == [run.digest]  # type: ignore[union-attr]
    assert "operator note" not in path.model_dump_json()  # type: ignore[union-attr]


def test_a_crash_between_execution_and_claim_leaves_no_referenced_attempt_and_a_retry_converges(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    added = world.act(OWNER, world.root, "add-1", ADD)

    def power_lost(*_: object) -> None:
        raise OSError("power lost after the attempt directory was staged")

    with monkeypatch.context() as patch:
        patch.setattr(WorkbenchStore, "_claim", power_lost)
        with pytest.raises(OSError, match="power lost"):
            world.act(OWNER, added.digest, "refit-1", REFINE)
    orphans = {p.name for p in (world.store.root / "attempts").iterdir()} - {world.root, added.digest}

    assert len(orphans) == 1
    assert world.tree() == [world.root, added.digest]
    with pytest.raises(AttemptAccessError):
        world.store.view(OWNER, orphans.pop())

    retried = world.act(OWNER, added.digest, "refit-1", REFINE)
    again = world.act(OWNER, added.digest, "refit-1", REFINE)

    assert again.digest == retried.digest
    assert len(world.refine.calls) == 2
    assert world.tree() == [world.root, added.digest, retried.digest]


def test_an_infrastructure_failure_is_an_invalid_execution_and_a_new_key_executes_again(
    world: World,
) -> None:
    added = world.act(OWNER, world.root, "add-1", ADD)
    world.refine.infra_failures = 1

    failed = world.act(OWNER, added.digest, "refit-1", REFINE)
    replayed = world.act(OWNER, added.digest, "refit-1", REFINE)
    retried = world.act(OWNER, added.digest, "refit-2", REFINE)

    assert isinstance(failed, RevealedAttemptView) and isinstance(retried, RevealedAttemptView)
    assert failed.attempt.outcome == InfraFailed(cause="executor_unavailable", detail="bgmn is not installed")
    assert failed.state.hypothesis is not None and failed.state.hypothesis.fit is None
    assert replayed.digest == failed.digest
    assert retried.digest != failed.digest
    assert retried.attempt.outcome.status == "completed"
    assert len(world.refine.calls) == 2


def test_an_import_is_idempotent_on_its_trial_and_seeds_the_hypothesis(world: World) -> None:
    receipt = b'{"verdict": "ACCEPT"}\n'
    action = ImportRun.model_validate(
        {
            "harbor": {
                "job_name": "u6-dev900",
                "trial_id": "task-dev900__abc",
                "task_sha256": "4" * 64,
                "pattern_sha256": CASE.pattern.sha256,
                "cif_sha256": {c.candidate_id: c.cif.sha256 for c in CASE.candidates if c.candidate_id in ("CIF000001", "CIF000003")},
            }
        }
    )
    result = ImportedRunResult.model_validate(
        {
            "agent": {"model_id": "qwen3-32b", "model_version": None, "harness": "harbor-local-mcp", "harness_version": "1"},
            "budget_wall_seconds": 1720,
            "wall_seconds": 900.0,
            "cost_usd": None,
            "end": {"end": "submitted", "answer": {"verdict": "ACCEPT", "phases": ["CIF000001", "CIF000003"]}},
            "basis_fit": None,
            "trajectory": [],
            "completeness": {"kind": "incomplete", "missing": ["native_lst"]},
        }
    )
    blobs = {BlobDigest(hashlib.sha256(receipt).hexdigest()): receipt}

    imported = world.store.import_run(IMPORTER, world.root, action, result, blobs)
    again = world.store.import_run(IMPORTER, world.root, action, result, blobs)
    with pytest.raises(AttemptAccessError):
        world.store.import_run(OWNER, world.root, action, result, blobs)
    foreign = action.model_copy(update={"harbor": action.harbor.model_copy(update={"pattern_sha256": "5" * 64})})
    with pytest.raises(ActionRefused, match="not this root's case"):
        world.store.import_run(IMPORTER, world.root, foreign, result, blobs)
    view = world.store.view(OWNER, imported)

    assert again == imported
    assert world.tree() == [world.root, imported]
    assert isinstance(view, RevealedAttemptView)
    assert view.state.hypothesis is not None
    assert view.state.hypothesis.phases == ("CIF000001", "CIF000003")
    assert view.state.hypothesis.fit_origin is None


def test_every_attempt_and_blob_the_store_wrote_hashes_to_its_name(world: World) -> None:
    world.store.grant(OWNER, world.root, "agent-1", (CaseOperation.INSPECT, CaseOperation.ACTION))
    refused = world.act(OWNER, world.root, "refine-empty", REFINE)
    added = world.act(OWNER, world.root, "add-1", ADD)
    world.act(OWNER, added.digest, "refit-1", REFINE)
    run = world.act(AGENT, added.digest, "run-1", RUN_START)
    world.act(AGENT, run.digest, "swap-1", {"verb": "replace_phase", "old": "CIF000002", "new": "CIF000003"})

    assert isinstance(refused, RevealedAttemptView)
    assert refused.attempt.outcome == Refused(reason_code="no_phases", detail="the phase set is empty")
    attempts = sorted((world.store.root / "attempts").iterdir())
    blobs = sorted((world.store.root / "blobs").iterdir())
    assert len(attempts) == 6 and len(blobs) == len(BLOBS) + 1
    for directory in attempts:
        assert hashlib.sha256((directory / "attempt.json").read_bytes()).hexdigest() == directory.name
    for blob in blobs:
        assert hashlib.sha256(blob.read_bytes()).hexdigest() == blob.name


def test_the_store_halts_an_agent_run_on_repeated_malformed_calls_and_on_a_model_mismatch(
    world: World,
) -> None:
    world.store.grant(OWNER, world.root, "agent-1", (CaseOperation.INSPECT, CaseOperation.ACTION))
    parent = world.act(AGENT, world.root, "run-1", RUN_START).digest
    for n in range(3):
        parent = world.act(AGENT, parent, f"bad-{n}", {"verb": "refine", "recipe": "fastest"}).digest
    halted = world.act(AGENT, parent, "edit-1", ADD)
    second_run = world.act(AGENT, world.root, "run-2", RUN_START)
    receipt = {"model_requested": "qwen3-32b", "model_observed": "qwen3-8b", "response_id": "r-1"}
    mismatched = world.act(AGENT, second_run.digest, "edit-2", ADD, completion=receipt)

    assert isinstance(halted, ActorAttemptView) and isinstance(mismatched, ActorAttemptView)
    assert halted.action == Halt(reason=MalformedToolUse(consecutive_rejections=3))
    assert mismatched.action == Halt(
        reason=InfraFailure(cause="model_identity_mismatch", detail="requested qwen3-32b, observed qwen3-8b")
    )
    with pytest.raises(ActionRefused, match="terminal"):
        world.act(AGENT, halted.digest, "edit-3", ADD)
    assert world.refine.calls == []


def test_an_accept_must_rest_on_a_rietveld_fit_on_its_own_path(world: World) -> None:
    world.store.grant(OWNER, world.root, "agent-1", (CaseOperation.INSPECT, CaseOperation.ACTION))
    added = world.act(OWNER, world.root, "add-1", ADD)
    refit = world.act(OWNER, added.digest, "refit-1", REFINE)
    run = world.act(AGENT, refit.digest, "run-1", RUN_START)
    uncertainty = {"identification": "two phases explain every peak", "quantification": "not assessed"}

    on_edit = world.act(
        AGENT, run.digest, "submit-1",
        {"verb": "submit", "answer": {"verdict": "ACCEPT", "basis": added.digest, "uncertainty": uncertainty}},
    )
    on_fit = world.act(
        AGENT, run.digest, "submit-2",
        {"verb": "submit", "answer": {"verdict": "ACCEPT", "basis": refit.digest, "uncertainty": uncertainty}},
    )

    assert isinstance(on_edit, ActorAttemptView) and isinstance(on_fit, ActorAttemptView)
    assert on_edit.outcome.message == f"basis_not_rietveld_ancestor: {added.digest}"
    assert on_fit.outcome.status == "recorded"
