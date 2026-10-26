"""The import and export commands, driven through cli.main on a synthetic lab case and run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from phaseeval.workbench import cli

REPO = Path(__file__).resolve().parents[2]
CASE = "LAB001"
PATTERN = "".join(f"{10 + i * 0.5:.2f} {100 + (i % 7) * 13}\n" for i in range(60)).encode()
CIF = b"data_NaCl\n_cell_length_a 5.64\n_symmetry_space_group_name_H-M 'F m -3 m'\n"
CIF_SHA = hashlib.sha256(CIF).hexdigest()
PATTERN_SHA = hashlib.sha256(PATTERN).hexdigest()
IPV4 = re.compile(rb"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def _row(locator: str) -> dict[str, str]:
    return {"class": "packaging", "locator": locator}


def build_case(root: Path, *, pattern_sha: str = PATTERN_SHA) -> Path:
    case = root / "case"
    (case / f"candidates/{CASE}").mkdir(parents=True)
    (case / f"{CASE}.xy").write_bytes(PATTERN)
    (case / f"candidates/{CASE}/nacl.cif").write_bytes(CIF)
    library = {
        "case_id": CASE,
        "candidates": [
            {
                "candidate_id": "nacl",
                "file": "nacl.cif",
                "sha256": CIF_SHA,
                "formula_sum": "Na Cl",
                "elements": ["Cl", "Na"],
                "source_library_id": "lab",
                "eligibility_status": "not_adjudicated",
            }
        ],
    }
    (case / f"candidates/{CASE}/manifest.json").write_text(json.dumps(library))
    sources = {
        p: _row(p)
        for p in (
            "measurement.instrument",
            "measurement.wavelength_angstrom",
            "instrument_profile.profile",
            "chemistry_elements",
            "material_context.specimen",
        )
    }
    manifest = {
        "task": "Identify the crystalline phases in this pattern.",
        "cases": [
            {
                "case_id": CASE,
                "pattern_file": f"{CASE}.xy",
                "pattern_sha256": pattern_sha,
                "field_sources": sources,
                "measurement": {
                    "instrument": "lab diffractometer",
                    "detector": "lab detector",
                    "wavelength_angstrom": 1.540598,
                    "kalpha2_wavelength_angstrom": 1.544426,
                },
                "instrument_profile": {"profile": "lab", "resolver_confidence": "matched", "warnings": []},
                "chemistry_elements": ["Cl", "Na"],
                "material_context": {"precursors": [], "processing": [], "specimen": "powder"},
            }
        ],
    }
    (case / "manifest.json").write_text(json.dumps(manifest))
    return case


def build_run(root: Path, *, phases: tuple[str, ...] = ("nacl",)) -> Path:
    run = root / "runs" / "r1"
    output = run / "artifacts/workspace/output"
    (output / "refine1").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"trials_dir": "jobs/lab-job", "agent": {"override_timeout_sec": 600}}))
    (run / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "lab-trial-1",
                "task_checksum": "0" * 64,
                "exception_info": None,
                "agent_info": {"name": "lab-agent", "version": "1", "model_info": {"name": "lab-model"}},
                "agent_execution": {"started_at": "2026-10-22T10:00:00+00:00", "finished_at": "2026-10-22T10:05:00+00:00"},
                "agent_result": {"cost_usd": None},
            }
        )
    )
    request = {"pattern_sha256": PATTERN_SHA, "candidates": [{"candidate_id": "nacl", "sha256": CIF_SHA}]}
    (output / "refine1/request.json").write_text(json.dumps(request))
    analysis = {"case_id": CASE, "phases": [{"candidate_id": p} for p in phases], "weight_fractions": None}
    (output / "analysis.json").write_text(json.dumps(analysis))
    return run


def run_cli(*argv: str) -> None:
    assert cli.main(list(argv)) == 0


def test_import_marks_a_user_case_and_the_run_lands_in_the_store(tmp_path: Path, capsys) -> None:
    case, run, store = build_case(tmp_path), build_run(tmp_path), tmp_path / "state/store"
    run_cli("import", "--case-dir", str(case), "--run-dir", str(run), "--case", CASE, "--store", str(store))
    out = capsys.readouterr().out
    assert "imported as a user case" in out and "no reference answer" in out

    assert not (tmp_path / "state/user-cases").exists()

    opened = cli.open_store(store, run.parent)
    (root,) = opened.roots(cli.OWNER)
    walked = opened.walk(root)
    assert walked[0][1].case.source.collection == "user_case"
    assert walked[0][1].case.split.split == "unassigned"
    assert len(walked) == 2  # the root and the imported run


def test_import_is_idempotent(tmp_path: Path) -> None:
    case, run, store = build_case(tmp_path), build_run(tmp_path), tmp_path / "state/store"
    for _ in range(2):
        run_cli("import", "--case-dir", str(case), "--run-dir", str(run), "--store", str(store))
    (root,) = cli.open_store(store, run.parent).roots(cli.OWNER)
    assert len(cli.open_store(store, run.parent).walk(root)) == 2


def test_a_pattern_hash_mismatch_is_refused(tmp_path: Path) -> None:
    case, run = build_case(tmp_path, pattern_sha="f" * 64), build_run(tmp_path)
    with pytest.raises(SystemExit, match="refused: .*does not hash"):
        cli.main(["import", "--case-dir", str(case), "--run-dir", str(run), "--store", str(tmp_path / "s")])


def test_a_phase_outside_the_library_is_refused(tmp_path: Path) -> None:
    case, run = build_case(tmp_path), build_run(tmp_path, phases=("quartz",))
    with pytest.raises(SystemExit, match="refused: .*not in LAB001's candidate library"):
        cli.main(["import", "--case-dir", str(case), "--run-dir", str(run), "--store", str(tmp_path / "s")])


def test_a_wrong_case_id_is_refused(tmp_path: Path) -> None:
    case, run = build_case(tmp_path), build_run(tmp_path)
    with pytest.raises(SystemExit, match="holds case LAB001, not OTHER"):
        cli.main(["import", "--case-dir", str(case), "--run-dir", str(run), "--case", "OTHER", "--store", str(tmp_path / "s")])


def test_export_validates_and_carries_no_local_paths_or_addresses(tmp_path: Path) -> None:
    case, run, store = build_case(tmp_path), build_run(tmp_path), tmp_path / "state/store"
    run_cli("import", "--case-dir", str(case), "--run-dir", str(run), "--store", str(store))
    out = tmp_path / "export"
    run_cli("export", "--out", str(out), "--store", str(store))

    validator = subprocess.run(
        [sys.executable, str(REPO / "scripts/workbench/validate_eval_dataset.py"), str(out)],
        capture_output=True,
        text=True,
        cwd=REPO / "scripts/workbench",
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert validator.returncode == 0, validator.stdout + validator.stderr
    assert len((out / "episodes.jsonl").read_text().splitlines()) == 1
    for path in out.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert b"/Users/" not in data and b"/home/" not in data and str(tmp_path).encode() not in data
            assert not IPV4.search(data), path.name


def harness_case(root: Path) -> tuple[Path, Path]:
    """harness/examples/dev006/input copied, and a minimal run that refined NiO against it."""
    case = root / "dev006"
    shutil.copytree(REPO / "harness/examples/dev006/input", case)
    entry = json.loads((case / "manifest.json").read_text())["cases"][0]
    nio = json.loads((case / "candidates/DEV006/manifest.json").read_text())["candidates"][0]
    run = build_run(root, phases=(nio["candidate_id"],))
    output = run / "artifacts/workspace/output"
    request = {"pattern_sha256": entry["pattern_sha256"], "candidates": [{"candidate_id": nio["candidate_id"], "sha256": nio["sha256"]}]}
    (output / "refine1/request.json").write_text(json.dumps(request))
    analysis = {"case_id": "DEV006", "phases": [{"candidate_id": nio["candidate_id"]}], "weight_fractions": None}
    (output / "analysis.json").write_text(json.dumps(analysis))
    return case, run


def test_a_harness_layout_case_imports_and_its_episode_exports_and_validates(tmp_path: Path, capsys) -> None:
    case, run = harness_case(tmp_path)
    before = {p.relative_to(case): p.read_bytes() for p in case.rglob("*") if p.is_file()}
    store = tmp_path / "state/store"
    run_cli("import", "--case-dir", str(case), "--run-dir", str(run), "--case", "DEV006", "--store", str(store))
    assert "imported as a user case" in capsys.readouterr().out
    assert {p.relative_to(case): p.read_bytes() for p in case.rglob("*") if p.is_file()} == before

    (root,) = cli.open_store(store, run.parent).roots(cli.OWNER)
    loaded = cli.open_store(store, run.parent).walk(root)[0][1].case
    assert {c.eligibility_status for c in loaded.candidates} == {"not_adjudicated"}
    assert loaded.measurement.profile.value.restricts_quantification

    out = tmp_path / "export"
    run_cli("export", "--out", str(out), "--store", str(store))
    validator = subprocess.run(
        [sys.executable, str(REPO / "scripts/workbench/validate_eval_dataset.py"), str(out)],
        capture_output=True,
        text=True,
        cwd=REPO / "scripts/workbench",
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert validator.returncode == 0, validator.stdout + validator.stderr
    (episode,) = (out / "episodes.jsonl").read_text().splitlines()
    assert "DEV006" in episode


def test_a_harness_layout_cif_hash_mismatch_is_refused(tmp_path: Path) -> None:
    case, run = harness_case(tmp_path)
    cif = next((case / "candidates/DEV006").glob("*.cif"))
    cif.write_bytes(cif.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="refused: .*does not hash"):
        cli.main(["import", "--case-dir", str(case), "--run-dir", str(run), "--store", str(tmp_path / "s")])


def test_an_imported_run_joins_the_raters_review_queue_and_its_review_opens(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from phaseeval.api.app import app
    from phaseeval.api.auth import PrincipalAuthenticator

    case, run = harness_case(tmp_path)
    store = tmp_path / "state/store"
    for _ in range(2):  # a re-import lists the snapshot once
        run_cli("import", "--case-dir", str(case), "--run-dir", str(run), "--case", "DEV006", "--store", str(store))
    queue = json.loads((store / "queues/owner-rater.json").read_text())
    assert queue["schema_version"] == "review-queue.v1" and queue["principal_id"] == "owner-rater"
    (digest,) = queue["snapshots"]

    token = "rater-token"
    principals = tmp_path / "principals.json"
    principals.write_text(
        json.dumps(
            {
                "schema_version": "axa.api_principals.v1",
                "principals": [
                    {
                        "principal_id": "owner-rater",
                        "principal_type": "user",
                        "roles": ["owner_rater"],
                        "credential_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    }
                ],
            }
        )
    )
    saved = dict(app.state._state)
    app.state.workbench_store = cli.open_store(store, run.parent)
    app.state.attempt_authenticator = PrincipalAuthenticator.from_file(principals)
    try:
        with TestClient(app) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            listed = client.get("/api/attempts/queue")
            assert listed.status_code == 200, listed.text
            assert [i["digest"] for i in listed.json()["items"]] == [digest]
            review = client.get(f"/api/attempts/snapshots/{digest}/review")
            assert review.status_code == 200, review.text
            assert review.json()["kind"] == "agent_answer"
    finally:
        app.state._state.clear()
        app.state._state.update(saved)


@pytest.mark.parametrize("command", ["import", "import-gpx", "import-bgmn"])
@pytest.mark.parametrize("existing", [False, True])
def test_a_refused_run_leaves_the_store_unchanged(tmp_path: Path, monkeypatch, command: str, existing: bool) -> None:
    from phaseeval.workbench.importers import gsasii

    case, run, store = build_case(tmp_path), build_run(tmp_path, phases=("quartz",)), tmp_path / "state/store"
    if existing:
        store.mkdir(parents=True)
        (store / "sentinel").write_bytes(b"existing store")
    before = {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}
    if command == "import-gpx":
        monkeypatch.delenv(gsasii.GSASII_SOURCE_VAR, raising=False)
        options = ["--project", str(tmp_path / "missing.gpx")]
    else:
        options = ["--run-dir", str(run)]
    with pytest.raises(SystemExit):
        cli.main([command, "--case-dir", str(case), *options, "--store", str(store)])
    assert {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")} == before


@pytest.mark.parametrize("existing", [False, True])
def test_a_missing_answer_receipt_leaves_the_store_unchanged(tmp_path: Path, existing: bool) -> None:
    case, run, store = build_case(tmp_path), build_run(tmp_path), tmp_path / "state/store"
    answer = run / "artifacts/workspace/output/analysis.json"
    outside = tmp_path / "analysis.json"
    answer.rename(outside)
    answer.symlink_to(outside)
    if existing:
        store.mkdir(parents=True)
        (store / "sentinel").write_bytes(b"existing store")
    before = {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}
    with pytest.raises(SystemExit, match="refused: .*answer file.*receipts"):
        cli.main(["import", "--case-dir", str(case), "--run-dir", str(run), "--store", str(store)])
    assert {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")} == before


def test_a_symlinked_case_ancestor_is_refused_before_any_write(tmp_path: Path) -> None:
    case, run = build_case(tmp_path / "real"), build_run(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(case.parent, target_is_directory=True)
    before = {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}
    refusal = None
    try:
        cli.main(["import", "--case-dir", str(alias / case.name), "--run-dir", str(run), "--store", str(tmp_path / "state/store")])
    except SystemExit as error:
        refusal = str(error)
    assert refusal is not None and "symlink" in refusal, "import accepted a symlinked case ancestor"
    assert {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")} == before


@pytest.mark.parametrize("component", ["file", "directory"])
def test_a_case_open_cannot_follow_a_substituted_symlink(tmp_path: Path, monkeypatch, component: str) -> None:
    case, run = build_case(tmp_path), build_run(tmp_path)
    outside = tmp_path / "outside"
    (outside / CASE).mkdir(parents=True)
    outside_cif = outside / CASE / "nacl.cif"
    outside_cif.write_bytes(CIF + b"# outside the case\n")
    sha = hashlib.sha256(outside_cif.read_bytes()).hexdigest()
    library_path = case / f"candidates/{CASE}/manifest.json"
    library = json.loads(library_path.read_bytes())
    library["candidates"][0]["sha256"] = sha
    library_path.write_text(json.dumps(library))
    request_path = run / "artifacts/workspace/output/refine1/request.json"
    request = json.loads(request_path.read_bytes())
    request["candidates"][0]["sha256"] = sha
    request_path.write_text(json.dumps(request))
    cif = case / f"candidates/{CASE}/nacl.cif"
    target = cif if component == "file" else case / "candidates"
    replacement = outside_cif if component == "file" else outside
    held = tmp_path / "held"
    path_open, os_open = Path.open, os.open
    substituted = False

    def substitute(open_file):
        nonlocal substituted
        substituted = True
        target.rename(held)
        target.symlink_to(replacement, target_is_directory=component == "directory")
        try:
            return open_file()
        finally:
            target.unlink()
            held.rename(target)

    def open_path(path, *args, **kwargs):
        if path == cif:
            return substitute(lambda: path_open(path, *args, **kwargs))
        return path_open(path, *args, **kwargs)

    def open_descriptor(path, *args, **kwargs):
        if path == target.name and kwargs.get("dir_fd") is not None:
            return substitute(lambda: os_open(path, *args, **kwargs))
        return os_open(path, *args, **kwargs)

    before = {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}
    monkeypatch.setattr(Path, "open", open_path)
    monkeypatch.setattr(os, "open", open_descriptor)
    refusal = None
    try:
        cli.main(["import", "--case-dir", str(case), "--run-dir", str(run), "--store", str(tmp_path / "state/store")])
    except SystemExit as error:
        refusal = str(error)
    monkeypatch.undo()
    assert substituted, "the open-time substitution did not run"
    assert refusal is not None and "symlink" in refusal, "import accepted CIF bytes through an open-time symlink"
    assert {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")} == before


def test_no_user_cases_directory_is_written(tmp_path: Path) -> None:
    case, run = harness_case(tmp_path)
    run_cli("import", "--case-dir", str(case), "--run-dir", str(run), "--store", str(tmp_path / "state/store"))
    assert not (tmp_path / "state/user-cases").exists()


@pytest.mark.parametrize("command", ["import-gpx", "import-bgmn"])
def test_an_external_binding_refusal_leaves_the_store_unchanged(tmp_path: Path, monkeypatch, command: str) -> None:
    from phaseeval.workbench.importers import gsasii

    case = build_case(tmp_path)
    store = tmp_path / "state/store"
    if command == "import-gpx":
        project = tmp_path / "project.gpx"
        project.write_bytes(b"project")
        analysis = gsasii.GpxAnalysis(
            source_sha256=hashlib.sha256(project.read_bytes()).hexdigest(),
            saved_using=None, reader_gsasii_version=None, rwp_percent=None, gof=None,
            converged=None, phases=(), histograms=(),
        )
        monkeypatch.setattr(gsasii, "read_gpx", lambda path: analysis)
        options = ["--project", str(project)]
    else:
        options = ["--run-dir", str(Path(__file__).parent / "fixtures/dev006_bgmn")]
    before = {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}
    with pytest.raises(SystemExit, match="refused"):
        cli.main([command, "--case-dir", str(case), *options, "--store", str(store)])
    assert {p.relative_to(tmp_path): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")} == before
