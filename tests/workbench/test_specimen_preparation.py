"""The optional preparation record of a case: validated where a case enters, kept in its digest
only when present, and shown in both briefs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_import_external import BUNDLE, CASE_DIR, _client, _queue, _restore, _validate_export, run_cli

from phaseeval.workbench import cli, importer
from phaseeval.workbench.importer import PackageError
from phaseeval.workbench.model import MaterialContext, SpecimenPreparation, canonical_bytes

ROOT = Path(__file__).resolve().parents[2]
PHYSICS = ROOT / "harness/physics"
SCRIPT = ROOT / "src/phaseeval/api/assets/workbench.js"
LIMIT = "Missing fields limit intensities and weight fractions, not peak positions alone."
RECORD = {"mount": "back-loaded", "grinding": "McCrone mill", "particle_diameter_um": 5, "spinning": True, "blank_scan": False}
# The digest of DEV006 with the task below, computed at the commit before this field existed. The
# case takes its task from the manifest here because a user case otherwise reads harness/agent/task.md.
TASK = "Identify the crystalline phases."
DEV006_CASE_SHA256 = "5f5d31ace5b1d041fc9533ee976e09651c1e8a005f2e618963b903bf1f2da2cb"


def _case_dir(tmp_path: Path, preparation: object = None) -> Path:
    case = tmp_path / "case"
    shutil.copytree(CASE_DIR, case)
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["task"] = TASK
    if preparation is not None:
        manifest["cases"][0]["material_context"]["specimen_preparation"] = preparation
    (case / "manifest.json").write_text(json.dumps(manifest))
    return case


def _load(tmp_path: Path, case: Path):
    records = tmp_path / "records"
    records.mkdir()
    labels, custody = cli.user_case_records(case, "DEV006", records)
    package = cli.harness_case_package(case, "DEV006", records / "package")
    return importer.read_case_package(package, labels, custody=custody)[0]


def test_a_case_without_the_block_keeps_its_digest_and_writes_no_key(tmp_path: Path) -> None:
    case = _load(tmp_path, _case_dir(tmp_path))
    assert case.material.specimen_preparation is None
    assert "specimen_preparation" not in case.material.model_dump(mode="json")
    assert b"specimen_preparation" not in canonical_bytes(case)
    assert hashlib.sha256(canonical_bytes(case)).hexdigest() == DEV006_CASE_SHA256


def test_no_shipped_case_declares_a_preparation_record() -> None:
    assert "specimen_preparation" not in json.loads((CASE_DIR / "manifest.json").read_text())["cases"][0]["material_context"]


def test_a_case_with_the_block_carries_it_with_its_source(tmp_path: Path) -> None:
    case = _load(tmp_path, _case_dir(tmp_path, RECORD))
    sourced = case.material.specimen_preparation
    assert sourced.value == SpecimenPreparation(mount="back-loaded", grinding="McCrone mill", particle_diameter_um=5.0, spinning=True, blank_scan=False)
    assert sourced.source.locator.endswith("material_context.specimen_preparation")
    round_trip = MaterialContext.model_validate(json.loads(json.dumps(case.material.model_dump(mode="json"))))
    assert round_trip == case.material


def test_every_field_is_nullable_and_the_free_text_specimen_stays_separate(tmp_path: Path) -> None:
    case = _load(tmp_path, _case_dir(tmp_path, {"mount": None, "grinding": "hand mortar"}))
    assert case.material.specimen_preparation.value == SpecimenPreparation(grinding="hand mortar")
    assert case.material.specimen is None


@pytest.mark.parametrize(
    "bad",
    [
        {"mount": "back-loaded", "operator": "someone"},
        {"particle_diameter_um": 0},
        {"particle_diameter_um": -2.0},
        {"particle_diameter_um": "5"},
        {"particle_diameter_um": True},
        {"spinning": "yes"},
        {"spinning": 1},
        {"blank_scan": 0},
        {"mount": 3},
        ["back-loaded"],
        "back-loaded",
    ],
)
def test_the_importer_refuses_a_malformed_record(tmp_path: Path, bad: object) -> None:
    with pytest.raises(PackageError, match="specimen_preparation"):
        _load(tmp_path, _case_dir(tmp_path, bad))


def test_the_record_reaches_the_review_brief_and_the_export_validates(tmp_path: Path) -> None:
    store = tmp_path / "state/store"
    run_cli("import-bgmn", "--case-dir", str(_case_dir(tmp_path, RECORD)), "--run-dir", str(BUNDLE), "--case", "DEV006", "--store", str(store))
    (digest,) = _queue(store)
    client, saved = _client(tmp_path, store)
    try:
        with client:
            review = client.get(f"/api/attempts/snapshots/{digest}/review")
            assert review.status_code == 200, review.text
    finally:
        _restore(saved)
    assert review.json()["brief"]["specimen_preparation"] == RECORD
    episodes = _validate_export(store, tmp_path / "export")
    assert '"specimen_preparation"' in episodes and "McCrone mill" in episodes


def test_a_case_without_the_block_shows_every_field_empty_in_the_review_brief(tmp_path: Path) -> None:
    store = tmp_path / "state/store"
    run_cli("import-bgmn", "--case-dir", str(_case_dir(tmp_path)), "--run-dir", str(BUNDLE), "--case", "DEV006", "--store", str(store))
    (digest,) = _queue(store)
    client, saved = _client(tmp_path, store)
    try:
        with client:
            review = client.get(f"/api/attempts/snapshots/{digest}/review")
    finally:
        _restore(saved)
    assert review.json()["brief"]["specimen_preparation"] == dict.fromkeys(RECORD)
    assert '"specimen_preparation"' not in _validate_export(store, tmp_path / "export")


@pytest.fixture
def case_brief(monkeypatch):
    monkeypatch.syspath_prepend(str(PHYSICS))
    spec = importlib.util.spec_from_file_location("case_brief_preparation", PHYSICS / "case_brief.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_tool_prints_each_field_or_not_recorded_then_the_limit(case_brief) -> None:
    lines = case_brief.preparation_lines({"mount": "back-loaded", "spinning": False})
    assert lines == [
        "Specimen preparation:",
        "  mount: back-loaded",
        "  grinding: not recorded",
        "  particle_diameter_um: not recorded",
        "  spinning: no",
        "  blank_scan: not recorded",
        f"  {LIMIT}",
    ]
    assert case_brief.preparation_lines(None).count("  mount: not recorded") == 1


def test_a_complete_record_has_no_limit_line(case_brief) -> None:
    lines = case_brief.preparation_lines(RECORD)
    assert LIMIT not in " ".join(lines) and "  particle_diameter_um: 5" in lines and "  spinning: yes" in lines


def test_the_tool_prints_it_once_for_dev006(case_brief, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["case_brief.py", "--input-dir", str(CASE_DIR), "--case", "DEV006"])
    case_brief.main()
    out = capsys.readouterr().out
    assert out.count("Specimen preparation:") == 1 and out.count(LIMIT) == 1 and "specimen_preparation" not in out


PAGE = """
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const ctx = vm.createContext({ console });
vm.runInContext(src.slice(0, src.indexOf('\\n$("auth").addEventListener')), ctx);
ctx.record = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(vm.runInContext(`[
  briefBar({ case_id: "DEV006", brief: {} }),
  briefBar({ case_id: "DEV006", brief: { specimen_preparation: record } }),
  briefBar({ case_id: "DEV006", brief: { specimen_preparation: { mount: "<b>x</b>" } } }),
]`, ctx)));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_brief_shows_each_field_and_the_limit_only_when_one_is_missing(tmp_path: Path) -> None:
    (tmp_path / "page.js").write_text(PAGE)
    run = subprocess.run(["node", str(tmp_path / "page.js"), str(SCRIPT), json.dumps(RECORD)], capture_output=True, text=True, timeout=60, check=True)
    empty, full, hostile = json.loads(run.stdout)
    assert empty.count("<b>not recorded</b>") >= 5 and LIMIT in empty
    assert "mount <b>back-loaded</b>" in full and "particle diameter <b>5 µm</b>" in full
    assert "spinning <b>yes</b>" in full and "blank scan <b>no</b>" in full and LIMIT not in full
    assert "&lt;b&gt;x&lt;/b&gt;" in hostile
