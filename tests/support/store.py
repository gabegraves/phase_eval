"""Synthetic cases, stores and authenticated review benches shared by the tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from phaseeval.api.auth import CaseOperation, Principal, PrincipalAuthenticator, PrincipalType
from phaseeval.workbench.labels import StageProjection, LabelStore
from phaseeval.workbench.model import BlobDigest, CaseManifest, ImportRun, ImportedRunResult
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.store import ActionRequest, AttemptView, WorkbenchStore
from phaseeval.api.app import create_app
from phaseeval.workbench.cli import Settings

from phaseeval.workbench import snapshot as snapshots
from tests.support.fake_bgmn import FakeRefine, recorded_fit


app = create_app(Settings.from_env({}))

OWNER = Principal("owner-1", PrincipalType.USER, frozenset({"owner"}))
AGENT = Principal("agent-1", PrincipalType.AGENT, frozenset())
IMPORTER = Principal("importer-1", PrincipalType.SERVICE, frozenset({"importer"}))

PATTERN = b"10.00 120\n10.02 131\n10.04 118\n"
CIFS = {
    "CIF000001": (b"data_Li2CO3\n", "Li2CO3", ["Li", "C", "O"]),
    "CIF000002": (b"data_Li2O\n", "Li2O", ["Li", "O"]),
    "CIF000003": (b"data_LiOH\n", "LiOH", ["Li", "O", "H"]),
}
BLOBS = {
    BlobDigest(hashlib.sha256(data).hexdigest()): data
    for data in (PATTERN, *(cif for cif, _, _ in CIFS.values()))
}
ADD = {"verb": "add_phases", "candidates": ["CIF000001", "CIF000002"]}
REFINE = {"verb": "refine"}
RUN_START = {
    "verb": "run_start",
    "harness": {
        "name": "harbor-local-mcp",
        "version": "1",
        "git_sha": "abc123",
        "instructions_sha256": "3" * 64,
    },
    "model_requested": "qwen3-32b",
    "budget": {"max_model_actions": 20, "max_wall_seconds": 3600, "max_executor_seconds": 1800},
    "preflight": None,
    "harbor": None,
}


def _ref(data: bytes, media_type: str, role: str) -> dict:
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "media_type": media_type,
        "role": role,
    }


def _source(channel: str) -> dict:
    adapter = {"adapter_id": "fixture", "version": "1"}
    return {"kind": "document", "channel": channel, "adapter": adapter, "document": None}


CASE_JSON = {
    "case_id": "DEV900",
    "source": {"kind": "experimental", "collection": "fixture_lab", "citation": "test fixture"},
    "exposure": "development_exposed",
    "split": {"split": "unassigned", "manifest_sha256": None, "chemical_system": None},
    "task_kind": "identification",
    "task": "Identify the crystalline phases.",
    "pattern": _ref(PATTERN, "application/vnd.axa.pxrd.xy", "observed_pattern"),
    "measurement": {
        "instrument": {
            "value": {
                "vendor": "Malvern Panalytical",
                "model": "Aeris",
                "geometry": "bragg_brentano",
                "detector": "PIXcel1D-Medipix3",
                "source_format": "xy",
            },
            "source": _source("instrument_export"),
        },
        "radiation": {"value": {"kind": "tube", "anode": "Cu"}, "source": _source("instrument_export")},
        "wavelength_provenance": None,
        "profile": {
            "value": {
                "instrument_profile": "Aeris-fds-Pixcel1d-Medipix3",
                "resolver_confidence": "matched",
                "reason_codes": [],
            },
            "source": _source("instrument_registry"),
        },
        "two_theta_deg": [10.0, 80.0],
    },
    "material": {
        "elements": {"value": ["Li", "C", "O"], "source": _source("eln")},
        "elements_basis": "declared",
        "precursors": [
            {
                "formula": {"value": "Li2CO3", "source": _source("synthesis_recipe")},
                "amount": {"value": {"value": 1.5, "unit": "g"}, "source": _source("custody_record")},
            }
        ],
        "processing": [],
        "declared_target": {"value": "Li2O", "source": _source("operator_note")},
    },
    "candidates": [
        {
            "candidate_id": candidate_id,
            "formula_sum": formula,
            "elements": elements,
            "cif": _ref(cif, "chemical/x-cif", "candidate_cif"),
            "source_library_id": f"COD {index}",
            "eligibility_status": "not_adjudicated",
        }
        for index, (candidate_id, (cif, formula, elements)) in enumerate(CIFS.items())
    ],
    "ingest": {
        "instrument_adapter": {"adapter_id": "xy", "version": "1"},
        "context_adapters": [],
        "candidate_selection": {"adapter_id": "fixture", "version": "1"},
        "sources": [],
    },
}
MANIFEST = json.dumps({"cases": [{
    "case_id": "DEV900",
    "chemistry_elements": ["Li", "C", "O"],
    "material_context": {"precursors": [{"formula": "Li2CO3"}]},
    "measurement": {},
    "instrument_profile": {},
}]}).encode()
CASE_JSON["ingest"]["sources"] = [{"original_name": "manifest.json", "blob": _ref(MANIFEST, "application/json", "raw_export")}]
BLOBS[BlobDigest(hashlib.sha256(MANIFEST).hexdigest())] = MANIFEST
CASE = CaseManifest.model_validate(CASE_JSON)


def _clock(start: datetime) -> Callable[[], datetime]:
    ticks: Iterator[int] = iter(range(10**6))
    return lambda: start + timedelta(seconds=next(ticks))


class World:
    def __init__(self, root: Path, start: datetime = datetime(2026, 10, 21, 16, tzinfo=UTC), *, refine=None) -> None:
        self.refine = refine or FakeRefine()
        self.store = WorkbenchStore(
            root,
            self.refine,
            public_seams(),
            _clock(start),
            import_root=root.parent / "runs",
            exposure=StageProjection(root),
        )
        self.root = self.store.load_case(OWNER, CASE, BLOBS)

    def act(self, principal: Principal, parent: str, key: str, action: dict, **request: object) -> AttemptView:
        return self.store.act(
            principal, parent, ActionRequest(idempotency_key=key, action=action, **request)
        )

    def tree(self) -> list[str]:
        return [digest for digest, _ in self.store.walk(self.root)]

    def files(self) -> list[str]:
        return sorted(str(p.relative_to(self.store.root)) for p in self.store.root.rglob("*"))


OWNER_RATER = Principal("owner-rater", PrincipalType.USER, frozenset({"owner_rater"}))

RATER_A = Principal("rater-a", PrincipalType.USER, frozenset({"rater"}))

RATER_B = Principal("rater-b", PrincipalType.USER, frozenset({"rater"}))

JUDGE = Principal("judge-opus", PrincipalType.MODEL_JUDGE, frozenset())

MIXED = Principal("owner-mixed", PrincipalType.USER, frozenset({"owner", "owner_rater"}))

PRINCIPALS = (OWNER, OWNER_RATER, MIXED, RATER_A, RATER_B, AGENT, IMPORTER, JUDGE)

BANNED = {"reference", "reference_match", "job", "model", "harness", "series", "episode_id", "siblings"}

ALLOW = {
    "participation_and_storage": True,
    "internal_evaluation_and_calibration": True,
    "public_release": False,
    "training": False,
    "public_identity_disclosure": False,
}

TRAJECTORY = [
    {"index": 0, "tool": "refine", "arguments_sha256": "6" * 64, "summary": "refine CIF000001,CIF000002"},
    {"index": 3, "tool": "refine", "arguments_sha256": "7" * 64, "summary": "refine CIF000001"},
]

def _headers(principal: Principal) -> dict[str, str]:
    return {"Authorization": f"Bearer {principal.principal_id}-secret"}

def _import(store: WorkbenchStore, root: str, trial: str, phases: tuple[str, ...]) -> str:
    receipt = json.dumps({"trial": trial}).encode()
    action = ImportRun.model_validate(
        {
            "harbor": {
                "job_name": "u6-dev900",
                "trial_id": trial,
                "task_sha256": "4" * 64,
                "pattern_sha256": CASE.pattern.sha256,
                "cif_sha256": {c.candidate_id: c.cif.sha256 for c in CASE.candidates if c.candidate_id in phases},
            }
        }
    )
    curve, fit_blobs = recorded_fit(phases, [50.0] * len(phases), rwp=3.5)
    result = ImportedRunResult.model_validate(
        {
            "agent": {"model_id": "qwen3-32b", "model_version": None, "harness": "harbor-local-mcp", "harness_version": "1"},
            "budget_wall_seconds": 1720,
            "wall_seconds": 900.0,
            "cost_usd": None,
            "end": {"end": "submitted", "answer": {"verdict": "ACCEPT", "phases": list(phases)}},
            "basis_fit": {
                "stage": "rietveld", "quantification": "quantitative", "rwp_percent": 3.5, "gof": 1.8,
                "phases": [{"candidate_id": p, "weight_percent": 50.0, "weight_percent_error": None,
                            "cell": dict.fromkeys("a b c alpha beta gamma".split())} for p in phases],
                "curve": _ref(curve, "text/csv", "fit_curve"),
            },
            "trajectory": TRAJECTORY,
            "completeness": {"kind": "incomplete", "missing": ["native_lst"]},
        }
    )
    blobs = {BlobDigest(hashlib.sha256(receipt).hexdigest()): receipt}
    return store.import_run(IMPORTER, root, action, result, blobs | fit_blobs)

def _snapshot(root: str, origin: str, phases: tuple[str, ...], kind: str = "agent_answer") -> snapshots.AnalysisSnapshot:
    return snapshots.AnalysisSnapshot(
        kind=kind,
        origin=origin,
        root=root,
        case_id=CASE.case_id,
        pattern_sha256=CASE.pattern.sha256,
        fit_curve_sha256=None,
        stage="rietveld",
        rwp_percent=3.5,
        gof=1.3,
        quantification="quantitative",
        phases=tuple(
            snapshots.SnapshotPhase(
                phase_id=p, formula=None, space_group=None, weight_percent=50.0, weight_percent_error=None
            )
            for p in phases
        ),
        residual_peaks=(
            snapshots.ResidualPeak(peak_id="p23.95", two_theta_deg=23.95, height_percent=3.18, residual_counts=491.0),
        ),
        checks=(),
        answered=True,
    )

class Bench:
    """One root, two imported agent answers frozen as snapshots, two assigned raters."""

    def __init__(self, tmp_path: Path) -> None:
        self.store = WorkbenchStore(
            tmp_path / "store",
            FakeRefine(),
            public_seams(),
            _clock(datetime(2026, 10, 22, 6, tzinfo=UTC)),
            import_root=tmp_path / "runs",
            exposure=StageProjection(tmp_path / "store"),
        )
        self.labels = LabelStore(self.store)
        self.root = self.store.load_case(OWNER, CASE, BLOBS)
        for rater in ("owner-rater", "owner-mixed", "rater-a", "rater-b"):
            self.store.grant(OWNER, self.root, rater, (CaseOperation.INSPECT,))
        self.runs = (
            _import(self.store, self.root, "task-dev900__qwen", ("CIF000001", "CIF000002")),
            _import(self.store, self.root, "task-dev900__astra", ("CIF000001", "CIF000003")),
        )
        self.snaps = (
            self.labels.put_snapshot(snapshots.build_snapshot(self.store, OWNER, self.runs[0], "agent_answer")),
            self.labels.put_snapshot(snapshots.build_snapshot(self.store, OWNER, self.runs[1], "agent_answer")),
        )
        self.reference = self.labels.put_snapshot(
            _snapshot(self.root, self.runs[0], ("CIF000001",), kind="reference")
        )
        registry = tmp_path / "principals.json"
        registry.write_text(
            json.dumps(
                {
                    "schema_version": "axa.api_principals.v1",
                    "principals": [
                        {
                            "principal_id": p.principal_id,
                            "principal_type": p.principal_type.value,
                            "credential_sha256": hashlib.sha256(f"{p.principal_id}-secret".encode()).hexdigest(),
                            "roles": sorted(p.roles),
                        }
                        for p in PRINCIPALS
                    ],
                }
            )
        )
        self.authenticator = PrincipalAuthenticator.from_file(registry)
        self.client = TestClient(app)

    def post(self, principal: Principal | None, path: str, body: dict[str, Any]) -> Any:
        headers = _headers(principal) if principal else {}
        return self.client.post(f"/api/attempts{path}", json=body, headers=headers)

    def get(self, principal: Principal | None, path: str) -> Any:
        return self.client.get(f"/api/attempts{path}", headers=_headers(principal) if principal else {})

    def assign(self, raters: list[str], *, snapshots: list[str] | None = None, principal: Principal = OWNER) -> Any:
        body = {"raters": raters, "snapshots": list(self.snaps) if snapshots is None else snapshots}
        return self.post(principal, f"/roots/{self.root}/assignment", body)

    def label(self, principal: Principal, snap: str, **body: Any) -> Any:
        body = {"item": "phase_supported", "phase_id": "CIF000001", "answer": "yes"} | body
        return self.post(principal, f"/snapshots/{snap}/labels", body)

    def sign_all(self, principal: Principal) -> None:
        for snap in self.snaps:
            response = self.post(principal, f"/snapshots/{snap}/signoff", {"verdict": "would_sign"})
            assert response.status_code == 201, response.text

    def allow(self, principal: Principal, **choices: bool) -> None:
        response = self.post(principal, "/permissions", ALLOW | choices)
        assert response.status_code == 201, response.text


def write_judge_label(labels, principal, label):
    from phaseeval.workbench.labels import JudgeLabelBody

    found = labels._snapshot(label.snapshot)
    labels._store.grant(OWNER, found.root, principal.principal_id, (CaseOperation.INSPECT,))
    labels.snapshot(principal, label.snapshot)
    identity = label.rater.judge
    manifest = labels.register_judge_run(OWNER, f'{identity.run_id}-{label.snapshot[:12]}', identity.model,
                                         identity.prompt_sha256, 'dataset:test', (label.snapshot,))
    item, digest = next(iter(manifest.input_sha256.items()))
    return labels.judge_label(principal, item, JudgeLabelBody(
        rubric_item=label.item, phase_id=label.phase_id, answer=label.answer, reason=label.reason,
        judge=identity.model_copy(update={'run_id': manifest.run_id, 'input_sha256': digest})))
