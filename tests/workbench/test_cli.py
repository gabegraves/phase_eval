"""The import and export commands, driven through cli.main on a synthetic lab case and run."""

from __future__ import annotations

import hashlib
import json
import os
import re
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


def test_import_writes_user_case_records_and_the_run_lands_in_the_store(tmp_path: Path, capsys) -> None:
    case, run, store = build_case(tmp_path), build_run(tmp_path), tmp_path / "state/store"
    run_cli("import", "--case-dir", str(case), "--run-dir", str(run), "--case", CASE, "--store", str(store))
    out = capsys.readouterr().out
    assert "imported as a user case" in out and "no reference answer" in out

    custody = json.loads((tmp_path / f"state/user-cases/{CASE}/custody.json").read_text())
    labels = json.loads((tmp_path / f"state/user-cases/{CASE}/case_labels.json").read_text())
    assert custody["reference_answer"] is None and custody["split_check"]["verdict"] == "unassigned"
    assert labels["cases"][0]["split"] == "unassigned"
    assert not any("reference" in key and value for key, value in labels["cases"][0].items())

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
