import hashlib
import json
import os
from pathlib import Path

import pytest

from phaseeval.workbench import executors
from phaseeval.workbench.evidence import U4_BOUNDS, compare_replay
from phaseeval.workbench.executors import (
    LIVE_BGMN_THREADS,
    bgmn_wavelength,
    read_lst_facts,
    read_recorded_execution,
    recipe_digest,
    refinement_config,
    run_recipe,
)
from phaseeval.workbench.model import (
    BackendIdentity,
    Cell,
    ExecutionRecord,
    MonochromaticRadiation,
    PhaseFit,
    PhaseInput,
    RefineResult,
    ReplayManifest,
    ResolvedRecipe,
    StageSummary,
    TubeRadiation,
)

REPO = Path(__file__).resolve().parents[2]
# U4 spike custody (git-excluded): inputs and the recorded runs with their preserved fit/.
SPIKE_U4 = Path(os.environ.get("AXA_SPIKE_U4", REPO / "data/custody/workbench/spike-u4"))
BUILD = "10cc9c1de0cca0eb6e12ff20abd44e959489c7d87082463639cfd297cfe9d23f"
PARSER = "5ab5c8197f5abdf2b7f16b7e398535cef583ef07"

requires_spike_custody = pytest.mark.skipif(
    not (SPIKE_U4 / "runs").is_dir(), reason=f"U4 spike custody not at {SPIKE_U4} (set AXA_SPIKE_U4)"
)


def _in_bgmn_slice() -> bool:
    cgroup = Path("/proc/self/cgroup")
    return cgroup.is_file() and "/bgmn.slice/" in cgroup.read_text()


def _recipe_digest(workdir: Path, n_threads: int) -> str:
    config = refinement_config(
        workdir,
        recipe="staged_rietveld",
        n_threads=n_threads,
        instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
        wavelength="Cu",
    )
    return recipe_digest(config.to_dict())


def test_recipe_digest_reproduces_the_recorded_agent_and_live_recipes(tmp_path: Path) -> None:
    assert _recipe_digest(tmp_path / "agent-run", 1) == "7ce17661e7c98fe7"
    assert _recipe_digest(tmp_path / "live-refit", LIVE_BGMN_THREADS) == "267a2ad2670edc56"


def test_a_cu_tube_is_the_lambda_symbol_and_only_a_monochromatic_beam_is_a_number() -> None:
    assert bgmn_wavelength(TubeRadiation(anode="Cu")) == "Cu"
    assert bgmn_wavelength(MonochromaticRadiation(wavelength_angstrom=1.540598)) == 1.540598


def _phase(candidate_id: str, weight: float | None, a: float) -> PhaseFit:
    cell = Cell(a=a, b=a, c=a, alpha=90.0, beta=90.0, gamma=90.0)
    return PhaseFit(candidate_id=candidate_id, weight_percent=weight, weight_percent_error=None, cell=cell)


def _stage(stage: str, rwp: float, rexp: float, phases: tuple[PhaseFit, ...], peaks: int) -> StageSummary:
    return StageSummary(
        stage=stage,  # type: ignore[arg-type]
        rwp_percent=rwp,
        gof=rwp / rexp,
        rexp_percent=rexp,
        converged=True,
        iterations=30,
        measured_points=8282,
        peaks=peaks,
        parameters=70,
        phases=phases,
        curve=None,
    )


def _record(
    order: tuple[str, ...] = ("CIF006427", "COD4031381"),
    *,
    final_weights: tuple[float, float] = (40.0, 60.0),
    binary_sha256: str | None = BUILD,
) -> ExecutionRecord:
    cif = {"CIF006427": "a" * 64, "COD4031381": "b" * 64}
    cells = {"CIF006427": 10.11787, "COD4031381": 6.53562}
    weights = dict(zip(("CIF006427", "COD4031381"), final_weights, strict=True))
    return ExecutionRecord(
        manifest=ReplayManifest(
            pattern_sha256="c" * 64,
            phases=tuple(PhaseInput(candidate_id=i, cif_sha256=cif[i]) for i in order),
            recipe=ResolvedRecipe(
                recipe="staged_rietveld",
                recipe_digest="7ce17661e7c98fe7",
                bgmn_threads=1,
                instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
                geq_sha256="d" * 64,
                ger_sha256="e" * 64,
                bgmn_wavelength="Cu",
            ),
            parser_revision=PARSER,
            backend=BackendIdentity(version="4.2.23", binary_sha256=binary_sha256),
            environment=None,
        ),
        result=RefineResult(
            stages=(
                _stage("pawley", 10.0, 3.47, tuple(_phase(i, None, cells[i]) for i in order), 207),
                _stage("rietveld", 6.62, 3.5, tuple(_phase(i, weights[i], cells[i]) for i in order), 174),
            ),
            final=None,
            refinement_failed=False,
            failure_reason=None,
        ),
    )


def test_the_same_phases_in_another_order_are_a_structural_mismatch_naming_the_order() -> None:
    findings = compare_replay(_record(), _record(("COD4031381", "CIF006427")), U4_BOUNDS)

    assert findings.verdict == "structural_mismatch"
    assert findings.structural_mismatches[0] == (
        "phases.order: recorded ['CIF006427', 'COD4031381'], live ['COD4031381', 'CIF006427']"
    )


def test_a_final_weight_004_points_away_is_outside_bounds_and_002_is_within() -> None:
    recorded = _record(final_weights=(40.0, 60.0))

    far = compare_replay(recorded, _record(final_weights=(40.04, 59.96)), U4_BOUNDS)
    near = compare_replay(recorded, _record(final_weights=(40.02, 59.98)), U4_BOUNDS)

    assert far.verdict == "outside_bounds"
    assert [d.quantity for d in far.deltas if not d.within] == [
        "weight_percent:CIF006427",
        "weight_percent:COD4031381",
    ]
    assert near.verdict == "within_bounds"


def test_a_recorded_run_without_its_binary_hash_is_incomplete_with_deltas_still_reported() -> None:
    findings = compare_replay(_record(binary_sha256=None), _record(final_weights=(40.04, 59.96)), U4_BOUNDS)

    assert findings.verdict == "incomplete"
    assert findings.missing == ("backend.binary_sha256",)
    assert [d.quantity for d in findings.deltas if not d.within] == [
        "weight_percent:CIF006427",
        "weight_percent:COD4031381",
    ]


@pytest.mark.parametrize("head", [None, "unavailable-commit"])
def test_a_recorded_parser_blob_does_not_need_the_original_git_history(tmp_path, monkeypatch, head) -> None:
    (tmp_path / "run.exec.json").write_text(json.dumps({"parser_revision": PARSER, "git_head": head}))
    monkeypatch.setattr(executors, "_git", lambda *args: pytest.fail("explicit revision must not query git"))

    assert executors._recorded_parser_revision(tmp_path / "run") == PARSER


@pytest.mark.parametrize("revision", [None, "", "a" * 39, "a" * 41, "A" * 40, "g" * 40, 123, []])
def test_a_malformed_explicit_parser_revision_cannot_fall_back_to_git(tmp_path, monkeypatch, revision) -> None:
    (tmp_path / "run.exec.json").write_text(json.dumps({"parser_revision": revision, "git_head": "legacy"}))
    monkeypatch.setattr(executors, "_git", lambda *args: pytest.fail("malformed revision must not query git"))

    with pytest.raises(ValueError, match="parser_revision"):
        executors._recorded_parser_revision(tmp_path / "run")


@pytest.mark.parametrize("revision", [PARSER, None])
def test_legacy_receipts_still_resolve_the_parser_from_git(tmp_path, monkeypatch, revision) -> None:
    (tmp_path / "run.exec.json").write_text(json.dumps({"git_head": "legacy"}))

    def git(*args):
        assert args == ("rev-parse", f"legacy:{executors.PARSER_PATH}")
        return revision

    monkeypatch.setattr(executors, "_git", git)
    assert executors._recorded_parser_revision(tmp_path / "run") == revision


@pytest.mark.parametrize("receipt", [None, {}, {"git_head": None}])
def test_a_run_without_parser_provenance_stays_unknown(tmp_path, monkeypatch, receipt) -> None:
    if receipt is not None:
        (tmp_path / "run.exec.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(executors, "_git", lambda *args: pytest.fail("missing provenance must not query git"))

    assert executors._recorded_parser_revision(tmp_path / "run") is None


@requires_spike_custody
def test_a_preserved_lst_gives_the_counts_and_weight_esd_the_parser_drops() -> None:
    fit = SPIKE_U4 / "runs/r1/dev006_corrected/fit"

    final = read_lst_facts((fit / "DEV006_rietveld.lst").read_text())
    pawley = read_lst_facts((fit / "DEV006_pawley.lst").read_text())

    assert (final.bgmn_version, final.measured_points, final.peaks, final.parameters) == ("4.2.23", 8282, 942, 95)
    assert dict(final.weight_esd) == {
        "CIF001783": 0.002,
        "CIF012075": 0.0035,
        "COD2310703": 0.0057,
        "CIF001772": 0.00055,
    }
    assert (pawley.measured_points, pawley.peaks, pawley.parameters) == (8282, 1072, 1107)
    assert dict(pawley.weight_esd) == {}


@requires_spike_custody
def test_a_recorded_spike_run_reads_back_its_recipe_parser_and_esd() -> None:
    record = read_recorded_execution(SPIKE_U4 / "runs/r1/dev006_corrected", binary_sha256=BUILD)

    assert [p.candidate_id for p in record.manifest.phases] == ["CIF001783", "CIF012075", "COD2310703", "CIF001772"]
    assert record.manifest.recipe.recipe_digest == "7ce17661e7c98fe7"
    assert record.manifest.parser_revision == PARSER
    assert record.manifest.backend == BackendIdentity(version="4.2.23", binary_sha256=BUILD)
    final = record.result.stages[-1]
    assert (final.stage, final.rwp_percent, final.rexp_percent) == ("rietveld", 2.5, 1.94)
    bi2co5 = final.phases[-1]
    assert (bi2co5.candidate_id, bi2co5.weight_percent, bi2co5.weight_percent_error) == (
        "CIF001772",
        2.7438330725294544,
        0.055,
    )


@pytest.mark.integration
@pytest.mark.slow
@requires_spike_custody
@pytest.mark.skipif(not _in_bgmn_slice(), reason="BGMN runs only inside bgmn.slice (CLAUDE.md, CPU Allocation)")
def test_the_public_dev001_control_re_executes_within_u4_bounds_of_its_spike_run(tmp_path: Path) -> None:
    recorded = read_recorded_execution(SPIKE_U4 / "runs/r1/dev001pub_cod_three_phase", binary_sha256=BUILD)
    inputs = SPIKE_U4 / "inputs/dev001-public"
    blobs = {hashlib.sha256(p.read_bytes()).hexdigest(): p.read_bytes() for p in inputs.rglob("*") if p.is_file()}

    live = run_recipe(
        recorded.manifest.pattern_sha256,
        recorded.manifest.phases,
        recorded.manifest.recipe,
        blobs.__getitem__,
        tmp_path / "dev001-control",
    )

    assert isinstance(live, ExecutionRecord), live
    findings = compare_replay(recorded, live, U4_BOUNDS)
    assert findings.verdict == "within_bounds", findings
    assert round(live.result.stages[-1].rwp_percent or 0.0, 2) == 6.62
