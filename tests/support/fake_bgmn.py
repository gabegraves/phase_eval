import hashlib

from phaseeval.workbench.executors import ExecutionContext, ExecutorResult
from phaseeval.workbench.model import BlobDigest, Completed, InfraFailed, ToolIdentity


def _ref(data: bytes, media_type: str, role: str) -> dict:
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "media_type": media_type,
        "role": role,
    }


FAKE_BGMN = {"name": "bgmn", "version": "fake", "binary_sha256": "3" * 64}


class FakeRefine:
    """Stands in for executors.refine: records each phase set it ran and returns a fit."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.infra_failures = 0

    def __call__(self, ctx: ExecutionContext, action: object) -> ExecutorResult:
        assert ctx.state.hypothesis is not None
        phases = ctx.state.hypothesis.phases
        self.calls.append(phases)
        tool = ToolIdentity(name="bgmn_refine", version="fake", axa_git_sha="0" * 40)
        pattern = ctx.read_blob(ctx.case.pattern.sha256)
        if self.infra_failures:
            self.infra_failures -= 1
            failed = InfraFailed(cause="executor_unavailable", detail="bgmn is not installed")
            return failed, tool, (), {}
        curve = b"two_theta,observed,calculated,difference\n10.00,120,119,1\n"
        curve_ref = _ref(curve, "text/csv", "fit_curve")
        cifs = {c.candidate_id: c.cif.sha256 for c in ctx.case.candidates}
        outcome = Completed.model_validate(
            {
                "result": {
                    "kind": "refine",
                    "stages": [],
                    "final": {
                        "stage": "rietveld",
                        "quantification": "quantitative",
                        "rwp_percent": 3.5,
                        "gof": 1.3,
                        "phases": [
                            {
                                "candidate_id": phase,
                                "weight_percent": 50.0,
                                "weight_percent_error": None,
                                "cell": dict.fromkeys(("a", "b", "c", "alpha", "beta", "gamma")),
                            }
                            for phase in phases
                        ],
                        "curve": curve_ref,
                    },
                    "refinement_failed": False,
                    "failure_reason": None,
                },
                "parent_state_sha256": "0" * 64,
                "artifacts": [curve_ref],
                "elapsed_seconds": 2.0,
                "custody_scan": "not_configured",
                "origin": {
                    "origin": "live",
                    "manifest": {
                        "pattern_sha256": hashlib.sha256(pattern).hexdigest(),
                        "phases": [{"candidate_id": p, "cif_sha256": cifs[p]} for p in phases],
                        "recipe": {
                            "recipe": "staged_rietveld",
                            "recipe_digest": "267a2ad2670edc56",
                            "bgmn_threads": 8,
                            "instrument_profile": "Aeris-fds-Pixcel1d-Medipix3",
                            "geq_sha256": "1" * 64,
                            "ger_sha256": "2" * 64,
                            "bgmn_wavelength": "Cu",
                        },
                        "parser_revision": "fake",
                        "backend": FAKE_BGMN,
                        "environment": {
                            "python_version": "3.12",
                            "os_release": "fixture",
                            "machine": "x86_64",
                            "axa_git_sha": "0" * 40,
                            "axa_source_dirty": False,
                            "bgmn": FAKE_BGMN,
                            "packages": [],
                            "threads": dict.fromkeys(
                                ("omp_num_threads", "openblas_num_threads", "mkl_num_threads")
                            ),
                        },
                    },
                },
            }
        )
        inputs = (ctx.case.pattern.sha256, *(cifs[p] for p in phases))
        return outcome, tool, inputs, {BlobDigest(curve_ref["sha256"]): curve}


def recorded_fit(phases, weights, rwp=4.1):
    import json
    import math

    rows = []
    for i in range(1001):
        angle = 10 + i * 0.05
        signal = 100000 * math.exp(-0.5 * ((angle - 30) / 0.1) ** 2)
        calculated = 1000 + signal
        observed = calculated + 3180 * math.exp(-0.5 * ((angle - 23.95) / 0.1) ** 2)
        rows.append((angle, observed, calculated, observed - calculated, 1000, signal))
    curve = ('two_theta,observed,calculated,difference,background\n' + ''.join(
        ','.join(map(str, row[:5])) + '\n' for row in rows)).encode()
    dia = ('TITEL=fixture/fit/DEV900_rietveld ' + ' '.join(
        f'STRUC[{i}]={phase}' for i, phase in enumerate(phases, 1)) + '\n' + ''.join(
        ' '.join(map(str, (row[0], row[1], row[2], row[4], *(row[5] / len(phases) for _ in phases)))) + '\n'
        for row in rows)).encode()
    result = json.dumps({
        'working_dir': 'fixture', 'refinement_failed': False,
        'pawley': {'rwp': 3.0, 'gof': 1.2, 'converged': True},
        'rietveld': {'rwp': rwp, 'gof': 1.8, 'converged': True, 'phases': [
            {'phase_id': phase, 'weight_fraction': weight, 'lattice_params': {}}
            for phase, weight in zip(phases, weights, strict=True)]},
    }).encode()
    return curve, {BlobDigest(hashlib.sha256(data).hexdigest()): data for data in (curve, dia, result)}
