"""Recorded run inputs and dataset exports shared by the tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from tests.support.store import write_judge_label
from tests.support.store import ADD, CASE, IMPORTER, OWNER, REFINE, RUN_START, World
from phaseeval.api.auth import CaseOperation, Principal, PrincipalType
from phaseeval.workbench import exports, snapshot
from phaseeval.workbench.exports import ExportInputs, RaterPseudonyms, build_dataset, write_dataset
from phaseeval.workbench.labels import (
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
from phaseeval.workbench.model import BlobDigest, ImportRun, ImportedRunResult
from phaseeval.workbench.policy import EligibilityLedger, RightsRegistry
from phaseeval.workbench.snapshot import AnalysisSnapshot
from tests.support.fake_bgmn import recorded_fit


WORKTREE = Path(__file__).resolve().parents[2]

NOW = datetime(2026, 10, 21, 17, tzinfo=UTC)

AGENT = Principal("agent-1", PrincipalType.AGENT, frozenset())

RATER = Principal("rater-ada", PrincipalType.USER, frozenset({"rater"}))

OWNER_RATER = Principal("owner-rater", PrincipalType.USER, frozenset({"owner_rater"}))


CURVE, FIT_BLOBS = recorded_fit(("CIF000001", "CIF000002"), (60.0, 40.0))

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
    if basis_fit is not None:
        blobs.update(FIT_BLOBS)
    return world.store.import_run(IMPORTER, world.root, action, result, blobs)

class Tree:
    """An answered import corrected by the owner, an import missing its basis fit, an
    unsuccessful import, and a host-side agent session."""

    def __init__(self, root: Path) -> None:
        self.world = World(root)
        world = self.world
        self.answered = import_run(world, "task-dev900__a", end=ANSWERED, completeness={"kind": "complete"}, basis_fit=BASIS_FIT)
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

def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]

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
            **({"review_payload": exports.blind_view} | options),
        )
        destination = out or self.out
        if destination.exists():
            shutil.rmtree(destination)
        write_dataset(dataset, destination)
        return dataset

    def label_both_kinds(self, digest: str) -> None:
        raters = (OWNER_RATER.principal_id, RATER.principal_id)
        self.labels.assign(OWNER, self.tree.world.root, raters, (digest,))
        for principal in (OWNER_RATER, RATER):
            self.labels.record_permission(principal, PERMIT)
            self.labels.label(principal, LabelRequest(snapshot=digest, item="fit_supports_conclusion", answer="yes"))
        write_judge_label(
            self.labels,
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
