"""Synthetic qualification cohorts shared by the API and CLI tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from tests.support.store import BLOBS, CASE, FakeRefine, OWNER, _clock
from phaseeval.api.auth import CaseOperation, Principal, PrincipalType
from phaseeval.workbench.labels import (
    ExposureLedger,
    LabelRequest,
    LabelStore,
    PermissionRequest,
    SignOffRequest,
)
from phaseeval.workbench.model import BlobDigest, ImportRun, ImportedRunResult
from phaseeval.workbench.qualification import Producer, QualificationPolicy, Qualifier
from phaseeval.workbench.seams import NoReference, public_seams
from phaseeval.workbench.snapshot import AnalysisSnapshot, SnapshotPhase, build_snapshot
from phaseeval.workbench.store import WorkbenchStore


def policy(**changes: object) -> QualificationPolicy:
    fields = {
        "version": "lab-a-2026q4",
        "justified_by": "hand-built test policy",
        "rater_kinds": ("expert", "owner"),
        "require_blind": False,
        "min_raters": 1,
        "phase_identity": "formula_and_space_group",
        "amount_abs_wt_pct": None,
        "expectation_rule": "lab-expectation-v2",
    }
    return QualificationPolicy(**{**fields, **changes})

RATER_A = Principal("rater-a", PrincipalType.USER, frozenset({"rater"}))
RATER_B = Principal("rater-b", PrincipalType.USER, frozenset({"rater"}))
IMPORTER = Principal("importer-1", PrincipalType.SERVICE, frozenset({"importer"}))
JUDGE_PRINCIPAL = Principal("judge-opus", PrincipalType.MODEL_JUDGE, frozenset())
CHOICES = {"public_release": False, "training": False, "public_identity_disclosure": False}
ALLOW = PermissionRequest(participation_and_storage=True, internal_evaluation_and_calibration=True, **CHOICES)
WITHDRAW = PermissionRequest(participation_and_storage=False, internal_evaluation_and_calibration=False, **CHOICES)
QWEN_V2 = Producer(model_id="qwen36-27b", harness="harness-v2", harness_version="2026-10-19")
ASTRA_V2 = Producer(model_id="gpt-6-astra", harness="harness-v2", harness_version="2026-10-19")
RIGHT, WRONG = ("CIF000001", "CIF000002"), ("CIF000001", "CIF000002", "CIF000003")
RULE = policy(rater_kinds=("expert",), min_raters=2, phase_identity="candidate_id")


def _import(store: WorkbenchStore, root: str, case, trial: str, model_id: str, phases: tuple[str, ...]) -> str:
    receipt = json.dumps({"trial": trial}).encode()
    action = ImportRun.model_validate(
        {
            "harbor": {
                "job_name": "u6-dev900",
                "trial_id": trial,
                "task_sha256": "4" * 64,
                "pattern_sha256": case.pattern.sha256,
                "cif_sha256": {c.candidate_id: c.cif.sha256 for c in case.candidates if c.candidate_id in phases},
            }
        }
    )
    agent = {"model_id": model_id, "model_version": None, "harness": "harness-v2", "harness_version": "2026-10-19"}
    result = ImportedRunResult.model_validate(
        {
            "agent": agent,
            "budget_wall_seconds": 3440,
            "wall_seconds": 900.0,
            "cost_usd": None,
            "end": {"end": "submitted", "answer": {"verdict": "ACCEPT", "phases": list(phases)}},
            "basis_fit": None,
            "trajectory": [],
            "completeness": {"kind": "incomplete", "missing": ["native_lst"]},
        }
    )
    blobs = {BlobDigest(hashlib.sha256(receipt).hexdigest()): receipt}
    return store.import_run(IMPORTER, root, action, result, blobs)


def _stored(root: str, case, origin: str, phases: tuple[str, ...], kind: str = "agent_answer") -> AnalysisSnapshot:
    return AnalysisSnapshot(
        kind=kind,
        origin=origin,
        root=root,
        case_id=case.case_id,
        pattern_sha256=case.pattern.sha256,
        fit_curve_sha256=None,
        stage="rietveld",
        rwp_percent=6.0,
        gof=1.3,
        quantification="restricted_profile",
        phases=tuple(
            SnapshotPhase(phase_id=p, formula=None, space_group=None, weight_percent=None, weight_percent_error=None)
            for p in phases
        ),
        residual_peaks=(),
        checks=(),
        answered=True,
    )


class Bench:
    """Two roots. On the first, raters A and B judge a qwen answer (signed) and an astra answer
    (rejected, CIF000003 unsupported); on the second only rater A judges a qwen answer."""

    def __init__(self, tmp_path) -> None:
        self.store = WorkbenchStore(
            tmp_path / "store",
            FakeRefine(),
            public_seams(),
            _clock(datetime(2026, 10, 22, 6, tzinfo=UTC)),
            import_root=tmp_path / "runs",
            exposure=ExposureLedger(tmp_path / "store"),
        )
        self.labels = LabelStore(self.store)
        other = CASE.model_copy(update={"case_id": "DEV901"})
        self.root = self.store.load_case(OWNER, CASE, BLOBS)
        self.other = self.store.load_case(OWNER, other, BLOBS)
        for rater in (RATER_A, RATER_B):
            for root in (self.root, self.other):
                self.store.grant(OWNER, root, rater.principal_id, (CaseOperation.INSPECT, CaseOperation.ACTION))
            self.labels.record_permission(rater, ALLOW)
        self.qwen = self._answer(self.root, CASE, "dev900__qwen", "qwen36-27b", RIGHT)
        self.astra = self._answer(self.root, CASE, "dev900__astra", "gpt-6-astra", WRONG)
        self.other_qwen = self._answer(self.other, other, "dev901__qwen", "qwen36-27b", RIGHT)
        self.labels.assign(OWNER, self.root, (RATER_A.principal_id, RATER_B.principal_id), (self.qwen, self.astra))
        self.labels.assign(OWNER, self.other, (RATER_A.principal_id,), (self.other_qwen,))
        for rater in (RATER_A, RATER_B):
            self._judge(rater, self.qwen, RIGHT, "would_sign")
            self._judge(rater, self.astra, WRONG, "would_not_sign")
        self._judge(RATER_A, self.other_qwen, RIGHT, "would_sign")
        self.qualifier = Qualifier(self.store, self.labels, NoReference())

    def _answer(self, root, case, trial, model_id, phases) -> str:
        origin = _import(self.store, root, case, trial, model_id, phases)
        return self.labels.put_snapshot(build_snapshot(self.store, OWNER, origin, "agent_answer"))

    def _judge(self, rater, snapshot, phases, verdict) -> None:
        for phase in phases:
            answer = "no" if phase == "CIF000003" else "yes"
            request = LabelRequest(snapshot=snapshot, item="phase_supported", phase_id=phase, answer=answer)
            self.labels.label(rater, request)
        self.labels.sign_off(rater, SignOffRequest(snapshot=snapshot, verdict=verdict))

    def freeze(self, name: str = "seam", seed: str | None = None) -> str:
        roots = (self.root, self.other)
        return self.qualifier.freeze(OWNER, lab="lab-a", name=name, roots=roots, policy=RULE, split_seed=seed)


