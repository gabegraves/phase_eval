from pathlib import Path

import pytest

from phaseeval.workbench.evidence import U4_BOUNDS, compare_replay
from phaseeval.workbench.executors import (
    LIVE_BGMN_THREADS,
    bgmn_wavelength,
    read_lst_facts,
    recipe_digest,
    refinement_config,
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

BUNDLE = Path(__file__).resolve().parent / 'fixtures/dev006_bgmn'
PAWLEY = Path(__file__).resolve().parent / 'fixtures/dev006_bgmn_pawley'
BUILD = "10cc9c1de0cca0eb6e12ff20abd44e959489c7d87082463639cfd297cfe9d23f"
PARSER = "5ab5c8197f5abdf2b7f16b7e398535cef583ef07"


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


def test_a_preserved_lst_gives_counts_and_weight_esd():
    final = read_lst_facts((BUNDLE / 'DEV006_rietveld.lst').read_text())
    pawley = read_lst_facts((PAWLEY / 'DEV006_pawley.lst').read_text())
    assert (final.bgmn_version, final.measured_points, final.peaks, final.parameters) == ('4.2.23', 8282, 461, 74)
    assert dict(final.weight_esd) == {'cod1010093': 0.0072, 'cod9008283': 0.011, 'cod1526458': 0.0042}
    assert (pawley.measured_points, pawley.peaks, pawley.parameters) == (8282, 510, 539)
    assert dict(pawley.weight_esd) == {}


def test_a_public_recorded_run_reads_back_its_recipe_counts_and_esd(tmp_path):
    import hashlib
    import json
    import shutil

    from phaseeval.workbench.executors import read_recorded_execution
    from phaseeval.workbench.importers.bgmn import read_bundle

    run = tmp_path / 'recorded'
    shutil.copytree(BUNDLE, run / 'fit')
    fit = read_bundle(BUNDLE)[0].fit
    config = refinement_config(run, recipe='staged_rietveld', n_threads=1,
                               instrument_profile='Aeris-fds-Pixcel1d-Medipix3', wavelength='Cu')
    request = {
        'config': config.to_dict(),
        'pattern_sha256': hashlib.sha256((BUNDLE / 'DEV006.xy').read_bytes()).hexdigest(),
        'candidates': [{'candidate_id': p.name, 'sha256': hashlib.sha256((BUNDLE / p.structure).read_bytes()).hexdigest()}
                       for p in fit.phases],
    }
    result = {
        'refinement_failed': False, 'failure_reason': None,
        'rietveld': {
            'rwp': fit.rwp.value, 'rexp': fit.rexp.value, 'converged': None, 'iteration_count': fit.iterations,
            'phases': [{'phase_id': p.name, 'weight_fraction': p.weight_percent, 'lattice_params': p.cell.model_dump()}
                       for p in fit.phases],
        },
    }
    (run / 'request.json').write_text(json.dumps(request))
    (run / 'result.json').write_text(json.dumps(result))

    record = read_recorded_execution(run, binary_sha256=None)

    assert record.manifest.recipe.recipe_digest == '7ce17661e7c98fe7'
    assert record.manifest.parser_revision is None and record.manifest.backend.binary_sha256 is None
    (stage,) = record.result.stages
    assert (stage.stage, stage.rwp_percent, stage.rexp_percent) == ('rietveld', 3.52, 1.94)
    assert (stage.measured_points, stage.peaks, stage.parameters) == (8282, 461, 74)
    assert [(p.candidate_id, p.weight_percent, p.weight_percent_error) for p in stage.phases] == [
        ('cod1010093', 35.72, 0.72), ('cod9008283', 43.3, 1.1), ('cod1526458', 21.0, 0.42),
    ]


@pytest.mark.integration
@pytest.mark.slow
def test_the_public_bgmn_control_reexecutes_within_recorded_bounds(tmp_path):
    import hashlib
    import shutil

    import pytest

    from phaseeval.refinement.bgmn.installer import BGMNInstaller
    from phaseeval.refinement.bgmn.runner import BGMNRunner
    from phaseeval.workbench.importers.bgmn import read_bundle

    installer = BGMNInstaller()
    if not installer.is_installed():
        pytest.skip('BGMN is not installed; the public control needs the native executable')
    binary_sha256 = hashlib.sha256(installer.get_bgmn_path().read_bytes()).hexdigest()
    if binary_sha256 != U4_BOUNDS.build.binary_sha256:
        pytest.skip('BGMN replay bounds require the recorded binary; this installation is a different build')
    recorded = read_bundle(BUNDLE)[0].fit
    workdir = tmp_path / 'live'
    shutil.copytree(BUNDLE, workdir)
    BGMNRunner(installer).run(workdir / 'DEV006_rietveld.sav', timeout_seconds=300)
    live = read_bundle(workdir)[0].fit
    assert live.rwp.value == pytest.approx(recorded.rwp.value, abs=U4_BOUNDS.rwp_rexp_abs_percent)
    assert tuple(p.weight_percent for p in live.phases) == pytest.approx(
        tuple(p.weight_percent for p in recorded.phases), abs=U4_BOUNDS.weight_abs_percent,
    )
