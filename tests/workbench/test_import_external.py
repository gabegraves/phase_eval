"""A refinement finished in GSAS-II, BGMN or Profex reaches the review queue like an agent run.

The BGMN path is driven through cli.main on the committed DEV006 Rietveld bundle and the DEV006
example case: the snapshot lands in the owner-rater's queue, the review route returns the pattern
and the curves, and the export validates. The GSAS-II mapping is driven from a GpxAnalysis record
through the same store, snapshot and route (no sandbox is needed for the mapping); the sandboxed
read itself is covered in test_importers_gsasii.py, and without bubblewrap import-gpx refuses.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from phaseeval.workbench import cli, snapshot
from phaseeval.workbench.importer import BindingError
from phaseeval.workbench.importers import external, gsasii
from phaseeval.workbench.importers.bgmn import BundleError, read_bundle
from phaseeval.workbench.exports import EpisodeRecord, validate_dataset
from phaseeval.workbench.labels import LabelStore
from phaseeval.workbench.model import ChildAttempt, ImportRun, ImportedRunResult, digest_of

REPO = Path(__file__).resolve().parents[2]
CASE_DIR = REPO / "harness/examples/dev006/input"
BUNDLE = Path(__file__).resolve().parent / "fixtures" / "dev006_bgmn"
# The same run's Pawley stage: its .sav, outputs and LeBail=1 .str files; the scan and the
# instrument files are the ones in BUNDLE.
PAWLEY_STAGE = Path(__file__).resolve().parent / "fixtures" / "dev006_bgmn_pawley"
CANDIDATES = ("cod_1010093", "cod_9008283", "cod_1526458")
RATER = "owner-rater"


def run_cli(*argv: str) -> None:
    assert cli.main(list(argv)) == 0


def _case(tmp_path: Path) -> Path:
    case = tmp_path / "dev006"
    shutil.copytree(CASE_DIR, case)
    return case


def _queue(store: Path) -> list[str]:
    return json.loads((store / f"queues/{RATER}.json").read_text())["snapshots"]


def _client(tmp_path: Path, store: Path):
    from fastapi.testclient import TestClient

    from phaseeval.api.app import app
    from phaseeval.api.auth import PrincipalAuthenticator

    token = "rater-token"
    principals = tmp_path / "principals.json"
    principals.write_text(
        json.dumps(
            {
                "schema_version": "axa.api_principals.v1",
                "principals": [
                    {
                        "principal_id": RATER,
                        "principal_type": "user",
                        "roles": ["owner_rater"],
                        "credential_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    }
                ],
            }
        )
    )
    saved = dict(app.state._state)
    app.state.workbench_store = cli.open_store(store, tmp_path)
    app.state.attempt_authenticator = PrincipalAuthenticator.from_file(principals)
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {token}"
    return client, saved


def _restore(saved: dict) -> None:
    from phaseeval.api.app import app

    app.state._state.clear()
    app.state._state.update(saved)


def _validate_export(store: Path, out: Path) -> str:
    run_cli("export", "--out", str(out), "--store", str(store))
    validator = subprocess.run(
        [sys.executable, str(REPO / "scripts/workbench/validate_eval_dataset.py"), str(out)],
        capture_output=True,
        text=True,
        cwd=REPO / "scripts/workbench",
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert validator.returncode == 0, validator.stdout + validator.stderr
    return (out / "episodes.jsonl").read_text()


def _imported_child(store: Path, tmp_path: Path) -> ChildAttempt:
    opened = cli.open_store(store, tmp_path)
    (root,) = opened.roots(cli.OWNER)
    walked = opened.walk(root)
    assert len(walked) == 2
    child = walked[1][1]
    assert isinstance(child, ChildAttempt)
    return child


def _review_has_the_fit(review: dict, weights: dict[str, float | None], stage: str = "rietveld") -> None:
    assert review["kind"] == "agent_answer" and review["case_id"] == "DEV006" and review["answered"]
    fit = review["fit"]
    assert fit["stage"] == stage and fit["points"] > 100
    for column in ("two_theta", "observed", "calculated", "difference"):
        assert len(fit[column]) == fit["points"]
    assert fit["phases"] == list(weights)
    answered = {p["candidate_id"]: p["weight_percent"] for p in review["answer"]["phases"]}
    assert answered == pytest.approx(weights)
    assert review["answer"]["verdict"] == "not recorded"
    checks = {c["code"]: c for c in review["checks"]}
    assert checks["K2"]["status"] in ("pass", "flag")
    if any(v is not None for v in weights.values()):
        assert checks["K3"]["status"] in ("pass", "flag")
    else:  # no amounts: K3's 2 wt% rule would pass over nothing, so it is not applicable
        assert checks["K3"]["status"] == "not_computable" and "no phase amounts" in checks["K3"]["detail"]
        assert [p["candidate_id"] for p in checks["K3"]["phases"]] == list(weights)  # the table still renders
    # The answer is the fit's own phase list, so K5 would compare a number with itself: not applicable.
    assert checks["K5"]["status"] == "not_computable" and "imported refinement" in checks["K5"]["detail"]
    forbidden = {"job", "model", "harness", "job_name", "model_id", "trial_id"}
    assert not forbidden & set(review) and not forbidden & set(review["answer"])


# --- BGMN and Profex, through the CLI ------------------------------------------------------------


def test_import_bgmn_queues_the_refinement_and_the_review_route_renders_it(tmp_path: Path, capsys) -> None:
    case, store = _case(tmp_path), tmp_path / "state/store"
    for _ in range(2):  # a re-import lists the snapshot once
        run_cli("import-bgmn", "--case-dir", str(case), "--run-dir", str(BUNDLE), "--case", "DEV006", "--store", str(store))
    out = capsys.readouterr().out
    assert "imported as a user case" in out and "source bgmn_saved_output (BGMN 4.2.23)" in out
    (digest,) = _queue(store)
    assert f"snapshot {digest} queued for {RATER}" in out

    child = _imported_child(store, tmp_path)
    assert isinstance(child.action, ImportRun) and child.action.harbor.job_name == external.BGMN_SAVED_OUTPUT
    assert child.action.external_source == external.BGMN_SAVED_OUTPUT  # the field the review keys on
    result = child.outcome.result
    assert isinstance(result, ImportedRunResult)
    assert (result.agent.harness, result.agent.model_id, result.agent.model_version) == (
        external.BGMN_SAVED_OUTPUT, "BGMN", "4.2.23",
    )
    assert result.trajectory == () and result.wall_seconds is None and result.cost_usd is None
    assert result.end.answer.verdict == "NOT_RECORDED" and result.end.answer.phases == CANDIDATES
    assert result.basis_fit is not None and result.basis_fit.rwp_percent == 3.52
    assert result.completeness.kind == "complete"
    assert child.action.harbor.cif_sha256.keys() == set(CANDIDATES)

    client, saved = _client(tmp_path, store)
    try:
        with client:
            listed = client.get("/api/attempts/queue")
            assert listed.status_code == 200, listed.text
            assert [i["digest"] for i in listed.json()["items"]] == [digest]
            review = client.get(f"/api/attempts/snapshots/{digest}/review")
            assert review.status_code == 200, review.text
    finally:
        _restore(saved)
    payload = review.json()
    _review_has_the_fit(payload, {"cod_1010093": 35.72, "cod_9008283": 43.3, "cod_1526458": 21.0})
    assert payload["fit"]["rwp"] == 3.52 and payload["fit"]["native_points"] == 8282
    assert set(payload["fit"]["phase_curves"]) == set(CANDIDATES)  # the .dia columns, by candidate
    esd = {p["candidate_id"]: p["esd"] for p in payload["answer"]["phases"]}
    assert esd == pytest.approx({"cod_1010093": 0.72, "cod_9008283": 1.1, "cod_1526458": 0.42})

    episodes = _validate_export(store, tmp_path / "export")
    (episode,) = episodes.splitlines()
    assert "DEV006" in episode and external.BGMN_SAVED_OUTPUT in episode


def test_a_stage_directory_with_several_controls_imports_the_named_one(tmp_path: Path) -> None:
    case, store = _case(tmp_path), tmp_path / "state/store"
    bundle = tmp_path / "fit"
    shutil.copytree(BUNDLE, bundle)
    shutil.copy(bundle / "DEV006_rietveld.sav", bundle / "DEV006_pawley.sav")
    with pytest.raises(SystemExit, match="refused: .*--control"):
        cli.main(["import-bgmn", "--case-dir", str(case), "--run-dir", str(bundle), "--store", str(store)])
    run_cli("import-bgmn", "--case-dir", str(case), "--run-dir", str(bundle), "--control", "DEV006_rietveld.sav", "--store", str(store))
    assert len(_queue(store)) == 1


def _pawley_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "fit"
    shutil.copytree(BUNDLE, bundle)
    for path in PAWLEY_STAGE.iterdir():
        shutil.copy(path, bundle / path.name)
    return bundle


def test_the_pawley_stage_of_a_run_is_shown_as_pawley_with_no_amounts(tmp_path: Path, capsys) -> None:
    """The harness's Pawley stage is a Le Bail extraction (LeBail=1 in every .str): the import
    reads that from the files, shows the stage as pawley, keeps its Rwp and drops the amounts BGMN
    printed for its fixed zero scales. The Rietveld control of the same directory stays rietveld."""
    case, store, bundle = _case(tmp_path), tmp_path / "state/store", _pawley_bundle(tmp_path)
    analysis, _ = read_bundle(bundle, control="DEV006_pawley.sav")
    assert analysis.fit.stage == "pawley" and "LeBail=1" in analysis.fit.stage_basis
    assert read_bundle(bundle, control="DEV006_rietveld.sav")[0].fit.stage == "rietveld"

    run_cli("import-bgmn", "--case-dir", str(case), "--run-dir", str(bundle), "--control", "DEV006_pawley.sav", "--store", str(store))
    out = capsys.readouterr().out
    assert "stage pawley; Rwp 3.31" in out and out.count("no weight fraction") == 3
    (digest,) = _queue(store)
    child = _imported_child(store, tmp_path)
    result = child.outcome.result
    assert result.basis_fit.stage == "pawley" and result.basis_fit.quantification == "not_quantitative"
    assert all(p.weight_percent is None for p in result.basis_fit.phases)
    assert result.end.answer.reported_weight_percent == {} and set(result.end.answer.phases) == set(CANDIDATES)
    # The one reason for the missing amounts is the stage, not BGMN's unprinted Q lines.
    gaps = [g for g in result.completeness.missing if g.startswith("weights:")]
    assert gaps == ["weights: a Pawley or Le Bail fit extracts intensities and has no phase amounts"]
    assert not any("prints no" in g for g in result.completeness.missing), result.completeness.missing
    assert "not recorded: weights: a Pawley or Le Bail fit" in out and out.count("not recorded: weights") == 1

    client, saved = _client(tmp_path, store)
    try:
        with client:
            review = client.get(f"/api/attempts/snapshots/{digest}/review")
            assert review.status_code == 200, review.text
    finally:
        _restore(saved)
    payload = review.json()
    _review_has_the_fit(payload, dict.fromkeys(result.end.answer.phases), stage="pawley")
    assert payload["fit"]["rwp"] == 3.31
    assert "Rietveld" not in payload["fit"].get("run_label", "")


def test_a_bundle_whose_stage_cannot_be_read_is_refused(tmp_path: Path) -> None:
    """Without the .str files the LeBail line cannot be read; the harness name of the control file
    still tells the stage, and a control file with neither is refused rather than called Rietveld."""
    bundle = tmp_path / "fit"
    shutil.copytree(BUNDLE, bundle)
    for path in bundle.glob("*.str"):
        path.unlink()
    analysis, _ = read_bundle(bundle)
    assert analysis.fit.stage == "rietveld" and "named for the rietveld stage" in analysis.fit.stage_basis
    (bundle / "DEV006_rietveld.sav").rename(bundle / "project.sav")
    with pytest.raises(BundleError, match="stage of project.sav cannot be determined.*LeBail"):
        read_bundle(bundle)


def test_a_bundle_whose_scan_is_not_the_cases_pattern_is_refused(tmp_path: Path) -> None:
    case, store = _case(tmp_path), tmp_path / "state/store"
    bundle = tmp_path / "fit"
    shutil.copytree(BUNDLE, bundle)
    scan = bundle / "DEV006.xy"
    lines = scan.read_text().splitlines()
    lines[10] = "10.118099 999999.000000"
    scan.write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="refused: scan 'DEV006.xy' is not this case's pattern"):
        cli.main(["import-bgmn", "--case-dir", str(case), "--run-dir", str(bundle), "--store", str(store)])
    assert not (store / f"queues/{RATER}.json").exists()


def test_a_phase_that_binds_to_no_candidate_is_refused_and_phase_binds_it(tmp_path: Path) -> None:
    case, store = _case(tmp_path), tmp_path / "state/store"
    library = json.loads((case / "candidates/DEV006/manifest.json").read_text())
    for record in library["candidates"]:
        if record["candidate_id"] == "cod_1010093":
            record["candidate_id"] = "nickel_oxide"
    (case / "candidates/DEV006/manifest.json").write_text(json.dumps(library))
    with pytest.raises(SystemExit, match=r"refused: no candidate of DEV006 for phase\(s\) \['cod1010093'\]"):
        cli.main(["import-bgmn", "--case-dir", str(case), "--run-dir", str(BUNDLE), "--store", str(store)])
    run_cli("import-bgmn", "--case-dir", str(case), "--run-dir", str(BUNDLE), "--store", str(store), "--phase", "cod1010093=nickel_oxide")
    child = _imported_child(store, tmp_path)
    assert child.outcome.result.end.answer.phases == ("nickel_oxide", "cod_9008283", "cod_1526458")


# --- GSAS-II ----------------------------------------------------------------------------------------


def _gpx_analysis(
    case_dir: Path, names: tuple[str, ...], fractions: tuple[float, ...], points: slice = slice(500, 2500)
) -> gsasii.GpxAnalysis:
    """A GpxAnalysis as gpx_reader would print it for a project that refined DEV006's scan: the
    observed points are the case's, the calculated and background series are smooth stand-ins."""
    rows = [tuple(map(float, line.split())) for line in (case_dir / "DEV006.xy").read_text().splitlines() if line.strip()]
    points = rows[points]
    x = tuple(p[0] for p in points)
    observed = tuple(p[1] for p in points)
    background = tuple(min(observed) for _ in points)
    calculated = tuple(0.5 * (o + b) for o, b in zip(observed, background, strict=True))
    cells = [{"length_a": 4.17, "length_b": 4.17, "length_c": 4.17, "angle_alpha": 90.0, "angle_beta": 90.0, "angle_gamma": 90.0, "volume": 72.5}] * len(names)
    return gsasii.GpxAnalysis(
        source_sha256="a" * 64,
        saved_using="#5837, v5.7.8 git 6b290a00",  # as the GSAS-II GUI records it
        reader_gsasii_version="5837",
        rwp_percent=7.25,
        gof=2.1,
        converged=True,
        phases=tuple(
            gsasii.GpxPhase(name=n, space_group="F m -3 m", cell=c, cell_refined=True, cell_esd={"length_a": 0.001}, pawley=False)
            for n, c in zip(names, cells, strict=True)
        ),
        histograms=(
            gsasii.GpxHistogram(
                name="PWDR DEV006.xy",
                instrument_type="PXC",
                x_quantity="two_theta_deg",
                wavelength_angstrom=1.5406,
                wr_percent=7.25,
                x=x,
                observed=observed,
                calculated=calculated,
                background=background,
                residual=tuple(o - c for o, c in zip(observed, calculated, strict=True)),
                fraction_basis="test",
                mass_fractions={
                    n: gsasii.GpxFraction(status="derived", scale_status="varied", value=f, su=0.01)
                    for n, f in zip(names, fractions, strict=True)
                },
                le_bail_phases=(),
            ),
        ),
    )


def test_a_gsasii_analysis_becomes_the_same_reviewable_snapshot(tmp_path: Path) -> None:
    case_dir, store_root = _case(tmp_path), tmp_path / "state/store"
    analysis = _gpx_analysis(case_dir, ("Nickel oxide", "cod_9008283"), (0.6, 0.4))
    project = b"the project bytes; never unpickled here"
    (tmp_path / "project.gpx").write_bytes(project)

    # The CLI's path: the case as a user case, the analysis mapped and written, the snapshot queued.
    args = cli.argparse.Namespace(case_dir=case_dir, case="DEV006", store=store_root)
    store_root, package, labels, custody, case_id = cli._case_inputs(args)
    store = cli.open_store(store_root, tmp_path)
    case, blobs = cli.importer.read_case_package(package, labels, custody=custody)
    root = store.load_case(cli.OWNER, case, blobs)
    with pytest.raises(BindingError, match="Nickel oxide"):
        external.from_gpx(analysis, project, case, blobs[case.pattern.sha256])
    action, result, run_blobs = external.from_gpx(
        analysis, project, case, blobs[case.pattern.sha256], {"Nickel oxide": "cod_1010093"}
    )
    assert action.harbor.job_name == external.GSASII_PROJECT and action.external_source == external.GSASII_PROJECT
    assert action.harbor.task_sha256 == hashlib.sha256(project).hexdigest()
    assert (result.agent.harness, result.agent.model_id, result.agent.model_version) == (
        external.GSASII_PROJECT, "GSAS-II", "5837 v5.7.8 6b290a00",  # the version tokens of the record only
    )
    assert result.end.answer.phases == ("cod_1010093", "cod_9008283")
    assert result.end.answer.reported_weight_percent == pytest.approx({"cod_1010093": 60.0, "cod_9008283": 40.0})
    assert result.basis_fit.rwp_percent == 7.25 and result.basis_fit.phases[0].weight_percent_error == pytest.approx(1.0)
    child = store.import_run(cli.IMPORTER, root, action, result, run_blobs)
    digest = cli.queue_for_review(store, root, child)
    assert _queue(store_root) == [digest]

    rater = cli.Principal(RATER, cli.PrincipalType.USER, frozenset({"owner_rater"}))
    payload = snapshot.blind_payload(store, LabelStore(store), rater, digest)
    _review_has_the_fit(payload, {"cod_1010093": 60.0, "cod_9008283": 40.0})
    assert payload["fit"]["native_points"] == 2000 and payload["fit"]["phase_curves"] == {}
    assert payload["fit"]["run_label"] == snapshot.IMPORTED_FIT_LABEL
    assert "fit.phase_curves.cod_1010093" in payload["missing"]  # GSAS-II gives no per-phase columns

    episodes = _validate_export(store_root, tmp_path / "export")
    assert external.GSASII_PROJECT in episodes


@pytest.mark.parametrize("how", ["doPawley", "LeBail"])
def test_a_pawley_or_le_bail_project_is_shown_as_pawley_with_no_amounts(tmp_path: Path, how: str) -> None:
    """What the project records decides the stage: a phase with doPawley, or a histogram link
    with LeBail, makes the fit Pawley-stage, and the fractions GSAS-II derives for it are dropped."""
    case_dir = _case(tmp_path)
    analysis = _gpx_analysis(case_dir, ("cod_1010093", "cod_9008283"), (0.6, 0.4))
    if how == "doPawley":
        phases = (analysis.phases[0].model_copy(update={"pawley": True}), analysis.phases[1])
        analysis = analysis.model_copy(update={"phases": phases})
    else:
        hist = analysis.histograms[0].model_copy(update={"le_bail_phases": ("cod_9008283",)})
        analysis = analysis.model_copy(update={"histograms": (hist,)})
    args = cli.argparse.Namespace(case_dir=case_dir, case="DEV006", store=tmp_path / "state/store")
    _, package, labels, custody, _ = cli._case_inputs(args)
    case, blobs = cli.importer.read_case_package(package, labels, custody=custody)
    _, result, _ = external.from_gpx(analysis, b"x", case, blobs[case.pattern.sha256])
    assert result.basis_fit.stage == "pawley" and result.basis_fit.rwp_percent == 7.25
    assert all(p.weight_percent is None and p.weight_percent_error is None for p in result.basis_fit.phases)
    assert result.end.answer.reported_weight_percent == {}
    assert any("Pawley or Le Bail" in gap for gap in result.completeness.missing)
    assert external.describe(result)["stage"] == "pawley"


def test_a_project_whose_histogram_is_not_the_cases_pattern_is_refused(tmp_path: Path) -> None:
    case_dir = _case(tmp_path)
    analysis = _gpx_analysis(case_dir, ("cod_1010093",), (1.0,))
    hist = analysis.histograms[0]
    shifted = hist.model_copy(update={"observed": tuple(v + 3.0 for v in hist.observed)})
    analysis = analysis.model_copy(update={"histograms": (shifted,)})
    args = cli.argparse.Namespace(case_dir=case_dir, case="DEV006", store=tmp_path / "state/store")
    _, package, labels, custody, _ = cli._case_inputs(args)
    case, blobs = cli.importer.read_case_package(package, labels, custody=custody)
    with pytest.raises(BindingError, match="not this case's pattern"):
        external.from_gpx(analysis, b"x", case, blobs[case.pattern.sha256])


def test_import_gpx_without_bubblewrap_refuses_and_queues_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gsasii.GSASII_SOURCE_VAR, str(tmp_path / "gsasii"))
    monkeypatch.setattr(gsasii.shutil, "which", lambda name: None)
    case, store = _case(tmp_path), tmp_path / "state/store"
    project = tmp_path / "project.gpx"
    project.write_bytes(b"any bytes")
    with pytest.raises(SystemExit, match="refused: a .gpx is opened only inside bubblewrap"):
        cli.main(["import-gpx", "--case-dir", str(case), "--project", str(project), "--case", "DEV006", "--store", str(store)])
    assert not (store / f"queues/{RATER}.json").exists()


HOSTILE_READER = Path(__file__).resolve().parent / "fixtures" / "exfiltrating_reader.py"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed; the sandbox runs only under it")
def test_a_reader_that_copies_what_it_can_read_leaks_nothing_to_the_store_payload_or_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """import-gpx with a reader that scrapes /proc and the environment for the importer's canary
    and copies all it read into its output: the import goes through (the output is a valid
    analysis of DEV006), and the canary is in no file of the store, not in the review payload and
    not in the export."""
    secret = f"secret-{hashlib.sha256(os.urandom(16)).hexdigest()[:24]}"
    monkeypatch.setenv("PHASEEVAL_TEST_CANARY", secret)
    (tmp_path / "gsasii").mkdir()
    monkeypatch.setenv(gsasii.GSASII_SOURCE_VAR, str(tmp_path / "gsasii"))
    monkeypatch.setattr(gsasii, "READER", HOSTILE_READER)
    case, store = _case(tmp_path), tmp_path / "state/store"
    rows = [line.split() for line in (case / "DEV006.xy").read_text().splitlines() if line.strip()][500:2500]
    project = tmp_path / "project.gpx"
    project.write_text(json.dumps({
        "x": [float(r[0]) for r in rows], "observed": [float(r[1]) for r in rows],
        "names": ["cod_1010093", "cod_9008283"], "fractions": [0.6, 0.4],
    }))

    run_cli("import-gpx", "--case-dir", str(case), "--project", str(project), "--case", "DEV006", "--store", str(store))
    out = capsys.readouterr().out
    assert "source gsasii_project (GSAS-II 5837)" in out and secret not in out
    (digest,) = _queue(store)

    stored = [p for p in store.rglob("*") if p.is_file()]
    assert stored and not [p for p in stored if secret.encode() in p.read_bytes()]
    opened = cli.open_store(store, tmp_path)
    rater = cli.Principal(RATER, cli.PrincipalType.USER, frozenset({"owner_rater"}))
    payload = snapshot.blind_payload(opened, LabelStore(opened), rater, digest)
    assert secret not in json.dumps(payload)
    _review_has_the_fit(payload, {"cod_1010093": 60.0, "cod_9008283": 40.0})
    episodes = _validate_export(store, tmp_path / "export")
    assert external.GSASII_PROJECT in episodes and secret not in episodes
    exported = [p for p in (tmp_path / "export").rglob("*") if p.is_file()]
    assert exported and not [p for p in exported if secret.encode() in p.read_bytes()]


def test_import_gpx_without_a_gsasii_source_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gsasii.GSASII_SOURCE_VAR, raising=False)
    case, store = _case(tmp_path), tmp_path / "state/store"
    project = tmp_path / "project.gpx"
    project.write_bytes(b"any bytes")
    with pytest.raises(SystemExit, match=gsasii.GSASII_SOURCE_VAR):
        cli.main(["import-gpx", "--case-dir", str(case), "--project", str(project), "--store", str(store)])


def test_the_external_handling_is_keyed_on_the_importers_field_not_on_the_job_name(tmp_path: Path) -> None:
    """An import whose job name is an external source's but that the external importer did not
    mark (external_source None, as every Harbor trial's import is) is reviewed as an agent run:
    its K5 is not the imported-refinement verdict. Only the field the importer sets selects it."""
    case_dir, store_root = _case(tmp_path), tmp_path / "state/store"
    args = cli.argparse.Namespace(case_dir=case_dir, case="DEV006", store=store_root)
    _, package, labels, custody, _ = cli._case_inputs(args)
    store = cli.open_store(store_root, tmp_path)
    case, blobs = cli.importer.read_case_package(package, labels, custody=custody)
    root = store.load_case(cli.OWNER, case, blobs)
    analysis, members = read_bundle(BUNDLE)
    action, result, run_blobs = external.from_bgmn(analysis, members, case, blobs[case.pattern.sha256])
    assert action.external_source == external.BGMN_SAVED_OUTPUT
    unmarked = action.model_copy(update={"external_source": None})
    assert unmarked.harbor.job_name == external.BGMN_SAVED_OUTPUT  # the run-supplied name alone
    child = store.import_run(cli.IMPORTER, root, unmarked, result, run_blobs)
    digest = cli.queue_for_review(store, root, child)
    rater = cli.Principal(RATER, cli.PrincipalType.USER, frozenset({"owner_rater"}))
    checks = {c["code"]: c for c in snapshot.blind_payload(store, LabelStore(store), rater, digest)["checks"]}
    assert "imported refinement" not in checks["K5"]["detail"] and checks["K5"]["detail"] == "No finished fit to check."
    assert ImportRun.model_validate(json.loads(unmarked.model_dump_json())).external_source is None


def test_read_bundle_and_from_bgmn_bind_by_the_processed_phase_name(tmp_path: Path) -> None:
    case_dir = _case(tmp_path)
    args = cli.argparse.Namespace(case_dir=case_dir, case="DEV006", store=tmp_path / "state/store")
    _, package, labels, custody, _ = cli._case_inputs(args)
    case, blobs = cli.importer.read_case_package(package, labels, custody=custody)
    analysis, members = read_bundle(BUNDLE)
    action, result, run_blobs = external.from_bgmn(analysis, members, case, blobs[case.pattern.sha256])
    assert result.end.answer.phases == CANDIDATES
    assert set(run_blobs) >= {m.ref.sha256 for m in analysis.members.values()}
    assert action.harbor.task_sha256 == analysis.members[analysis.control].ref.sha256


# --- records of agent runs are unchanged by the external_source field ------------------------------

DEMO_STORE = REPO / "launcher/demo/phaseeval-demo.tar.xz"


def _schema_before_external_source() -> dict:
    """The published episode schema as it was before ImportRun.external_source existed: the same
    schema without that property (byte-equal to the file at bbf1537), with additionalProperties
    false, so a record that writes the key is rejected by it."""
    schema = json.loads((REPO / "schemas/workbench-episode.v1.json").read_text())
    del schema["$defs"]["ImportRun"]["properties"]["external_source"]
    return schema


def test_agent_run_records_are_the_bytes_they_were_before_external_source(tmp_path: Path) -> None:
    """The bundled demo store was written before ImportRun had external_source. Read now: its
    export validates against the episode schema of that time, every exported step's digest is the
    digest of the attempt as serialized now (the store's invariant), and importing each run again
    with the stored action, result and receipts is the idempotent retry it was, not a conflict.
    Only an import the external importer marks writes the key."""
    with tarfile.open(DEMO_STORE, "r:xz") as tar:
        tar.extractall(tmp_path / "demo", filter="data")
    store_root, out = tmp_path / "demo/store", tmp_path / "export"
    run_cli("export", "--out", str(out), "--store", str(store_root))
    schemas = {p.name: json.loads(p.read_text()) for p in (REPO / "schemas").glob("*.json")}
    schemas["workbench-episode.v1.json"] = _schema_before_external_source()
    assert validate_dataset(out, schemas) == []
    episodes = [EpisodeRecord.model_validate_json(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    steps = [step for episode in episodes for step in episode.steps]
    assert len(steps) >= 6 and all(step.digest == digest_of(step.attempt) for step in steps)
    assert "external_source" not in (out / "episodes.jsonl").read_text()

    store = cli.open_store(store_root, store_root)
    retried = 0
    for root in store.roots(cli.OWNER):
        for digest, attempt in store.walk(root):
            if isinstance(attempt, ChildAttempt) and isinstance(attempt.action, ImportRun):
                assert attempt.action.external_source is None  # read back as unset
                assert b"external_source" not in attempt.action.model_dump_json().encode()
                receipts = {ref.sha256: store.read_blob(ref.sha256) for ref in attempt.outcome.origin.receipts}
                importer = cli.Principal(attempt.actor.principal_id, cli.PrincipalType.SERVICE, frozenset({"importer"}))
                assert store.import_run(importer, root, attempt.action, attempt.outcome.result, receipts) == digest
                retried += 1
    assert retried == 6

    marked = ImportRun(harbor=steps[0].attempt.action.harbor, external_source=external.GSASII_PROJECT)
    assert json.loads(marked.model_dump_json())["external_source"] == external.GSASII_PROJECT
    assert ImportRun.model_validate_json(marked.model_dump_json()) == marked


# --- a fit that explains nothing still queues --------------------------------------------------------


def test_a_fit_that_explains_no_peak_queues_with_a_bounded_k2_detail(tmp_path: Path) -> None:
    """A project saved before its phases were refined: the calculated curve is the background, so
    every peak of the scan is an unexplained residual peak. The K2 detail names the count and the
    strongest few, stays within SnapshotCheck's bound, and the import queues and exports."""
    case_dir, store_root = _case(tmp_path), tmp_path / "state/store"
    analysis = _gpx_analysis(case_dir, ("cod_1010093", "cod_9008283"), (0.6, 0.4), points=slice(None))
    hist = analysis.histograms[0]
    unrefined = hist.model_copy(update={"calculated": hist.background, "residual": tuple(o - b for o, b in zip(hist.observed, hist.background, strict=True))})
    analysis = analysis.model_copy(update={"histograms": (unrefined,), "rwp_percent": 61.0})
    args = cli.argparse.Namespace(case_dir=case_dir, case="DEV006", store=store_root)
    store_root, package, labels, custody, _ = cli._case_inputs(args)
    store = cli.open_store(store_root, tmp_path)
    case, blobs = cli.importer.read_case_package(package, labels, custody=custody)
    root = store.load_case(cli.OWNER, case, blobs)
    action, result, run_blobs = external.from_gpx(analysis, b"unrefined project", case, blobs[case.pattern.sha256])
    child = store.import_run(cli.IMPORTER, root, action, result, run_blobs)
    digest = cli.queue_for_review(store, root, child)
    assert _queue(store_root) == [digest]

    rater = cli.Principal(RATER, cli.PrincipalType.USER, frozenset({"owner_rater"}))
    checks = {c["code"]: c for c in snapshot.blind_payload(store, LabelStore(store), rater, digest)["checks"]}
    k2 = checks["K2"]
    assert k2["status"] == "flag" and len(k2["peaks"]) > snapshot.K2_DETAIL_PEAKS
    assert len(k2["detail"]) <= 4000
    assert k2["detail"].startswith(f"{len(k2['peaks'])} unexplained residual peaks at or above")
    assert f"the strongest {snapshot.K2_DETAIL_PEAKS}:" in k2["detail"] and "Method:" in k2["detail"]
    assert external.GSASII_PROJECT in _validate_export(store_root, tmp_path / "export")


# --- what the project says about who wrote it ------------------------------------------------------


@dataclass(frozen=True)
class _PreparedReader(gsasii.GpxSandbox):
    """read_gpx's real read loop, limits and validation, with a script that prints a prepared
    reader output in place of bwrap and the reader."""

    output: str = ""

    def argv(self, inbox: Path) -> list[str]:
        return [sys.executable, "-I", "-c", "import sys; sys.stdout.write(open(sys.argv[1]).read())", self.output]


def test_the_store_and_the_export_hold_only_the_version_of_what_wrote_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """import-gpx with a reader whose saved_using carries, beside GSAS-II's version record, the
    user, the host and a path: the attempt's model_version is the version tokens alone, and no
    file of the store or the export holds the rest."""
    user, host, path = "alice-hostile", "lab-pc-77", "/srv/alice-hostile/proj/x.gpx"
    case, store = _case(tmp_path), tmp_path / "state/store"
    payload = _gpx_analysis(case, ("cod_1010093", "cod_9008283"), (0.6, 0.4)).model_dump(mode="json")
    payload["saved_using"] = f"#5837, v5.7.8 git 6b290a00 saved by {user}@{host} from {path}"
    output = tmp_path / "reader-output.json"
    output.write_text(json.dumps(payload))
    prepared = _PreparedReader(python=Path(sys.executable), gsasii_source=tmp_path, output=str(output))
    monkeypatch.setattr(gsasii.GpxSandbox, "from_environment", classmethod(lambda cls, environ=None: prepared))
    project = tmp_path / "project.gpx"
    project.write_bytes(b"never unpickled here")

    run_cli("import-gpx", "--case-dir", str(case), "--project", str(project), "--case", "DEV006", "--store", str(store))
    out = capsys.readouterr().out
    assert "source gsasii_project (GSAS-II 5837 v5.7.8 6b290a00)" in out
    child = _imported_child(store, tmp_path)
    assert child.outcome.result.agent.model_version == "5837 v5.7.8 6b290a00"
    leaks = [token.encode() for token in (user, host, path)]
    stored = [p for p in store.rglob("*") if p.is_file()]
    assert stored and not [p for p in stored if any(leak in p.read_bytes() for leak in leaks)]
    episodes = _validate_export(store, tmp_path / "export")
    assert '"model_version":"5837 v5.7.8 6b290a00"' in episodes
    exported = [p for p in (tmp_path / "export").rglob("*") if p.is_file()]
    assert exported and not [p for p in exported if any(leak in p.read_bytes() for leak in leaks)]
