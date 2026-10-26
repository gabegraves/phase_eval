"""Case-folder trust boundaries, exercised through the import command and public reader."""

import json
import shutil
from pathlib import Path

import pytest

from phaseeval.workbench import cli, importer
from phaseeval.workbench.model import RootAttempt, digest_of

EXAMPLE = Path(__file__).resolve().parents[2] / "harness/examples/dev006/input"
DEV006_ROOT = "a243f16c7356e58c6c8add0adc95bb2f60e775fc16eab3fedbd48a7cb39995f3"


def _tree(root: Path) -> dict[str, bytes | None]:
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}


@pytest.mark.parametrize("attack", ["absolute_id", "dotdot_id", "dotdot_pattern", "absolute_pattern", "windows_pattern", "dot_pattern", "symlink_cif", "symlink_directory", "symlink_manifest", "outside_cif"])
def test_unsafe_case_paths_are_refused_before_any_write(tmp_path: Path, attack: str) -> None:
    case = tmp_path / "case"
    shutil.copytree(EXAMPLE, case)
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    entry = manifest["cases"][0]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("keep")
    if attack.endswith("_id"):
        entry["case_id"] = str(outside) if attack == "absolute_id" else ".."
    elif attack.endswith("pattern"):
        entry["pattern_file"] = {"dotdot_pattern": "../outside/sentinel", "absolute_pattern": str(outside / "sentinel"), "windows_pattern": "C:\\outside.xy", "dot_pattern": "./DEV006.xy"}[attack]
    elif attack == "outside_cif":
        path = case / "candidates/DEV006/manifest.json"
        library = json.loads(path.read_bytes())
        library["candidates"][0]["file"] = "../../../outside/sentinel"
        path.write_text(json.dumps(library))
    elif attack == "symlink_cif":
        cif = next(case.rglob("*.cif"))
        data = cif.read_bytes()
        cif.unlink()
        (outside / "candidate.cif").write_bytes(data)
        cif.symlink_to(outside / "candidate.cif")
    elif attack == "symlink_directory":
        shutil.move(case / "candidates", outside / "candidates")
        (case / "candidates").symlink_to(outside / "candidates", target_is_directory=True)
    manifest_path.write_text(json.dumps(manifest))
    if attack == "symlink_manifest":
        shutil.move(manifest_path, outside / "manifest.json")
        manifest_path.symlink_to(outside / "manifest.json")
    before = _tree(tmp_path)
    with pytest.raises(SystemExit, match="refused"):
        cli.main(["import", "--case-dir", str(case), "--run-dir", str(tmp_path / "missing-run"), "--store", str(tmp_path / "state/store")])
    assert _tree(tmp_path) == before


def test_the_dev006_root_digest_is_unchanged_for_both_layouts(tmp_path: Path) -> None:
    before = _tree(tmp_path)
    case, blobs = importer.case_manifest(importer.read_case(EXAMPLE))
    assert digest_of(RootAttempt(case=case)) == DEV006_ROOT
    assert _tree(tmp_path) == before
    package = tmp_path / "package"
    for source in case.ingest.sources:
        path = package / source.original_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blobs[source.blob.sha256])
    manifest = json.loads((package / "manifest.json").read_bytes())
    (package / manifest["cases"][0]["pattern_file"]).write_bytes(blobs[case.pattern.sha256])
    library = json.loads((package / "candidates/DEV006/manifest.json").read_bytes())
    for record in library["candidates"]:
        (package / "candidates/DEV006" / record["file"]).write_bytes(blobs[record["sha256"]])
    before = _tree(tmp_path)
    packaged, packaged_blobs = importer.case_manifest(importer.read_case(package))
    assert digest_of(RootAttempt(case=packaged)) == DEV006_ROOT
    assert packaged_blobs == blobs
    assert _tree(tmp_path) == before


def test_the_packaged_dev006_root_digest_is_unchanged(tmp_path: Path) -> None:
    import tarfile

    digest = "d51a3823f766972d0fab8d133e6485b00f181d5e857456b37965ada8c8899c61"
    with tarfile.open(EXAMPLE.parents[3] / "launcher/demo/phaseeval-demo.tar.xz") as archive:
        root = json.load(archive.extractfile(f"store/attempts/{digest}/attempt.json"))
        case = root["case"]
        for source in case["ingest"]["sources"]:
            path = tmp_path / source["original_name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(f"store/blobs/{source['blob']['sha256']}").read())
        manifest = json.loads((tmp_path / "manifest.json").read_bytes())
        pattern = manifest["cases"][0]["pattern_file"]
        (tmp_path / pattern).write_bytes(archive.extractfile(f"store/blobs/{case['pattern']['sha256']}").read())
        library = json.loads((tmp_path / "candidates/DEV006/manifest.json").read_bytes())
        for record in library["candidates"]:
            (tmp_path / "candidates/DEV006" / record["file"]).write_bytes(archive.extractfile(f"store/blobs/{record['sha256']}").read())
    label = {
        "case_id": "DEV006",
        "source": {"creators": "Fei, McDermott, Rom, Wang, Ceder", "dataset": "DARA benchmark (precursor mixtures)", "doi": "10.5281/zenodo.17410051", "license": "CC-BY-4.0"},
        "exposure": "development-exposed", "split": "dev", "chemical_system": "Bi-C-Li-Ni-O", "confirmation_tag": "confirmation",
    }
    (tmp_path / "case_labels.json").write_text(json.dumps({"cases": [label]}))
    (tmp_path / "custody.json").write_text(json.dumps({
        "case_id": "DEV006", "split_check": {"assigned_split": "dev", "verdict": "dev", "chemical_system": label["chemical_system"], "split_manifest_content_sha256": case["split"]["manifest_sha256"]},
        "labels": {"confirmation_tag": "confirmation", "exposure": "development-exposed"},
    }))
    before = _tree(tmp_path)
    loaded, _ = importer.case_manifest(importer.read_case(tmp_path))
    assert digest_of(RootAttempt(case=loaded)) == digest
    assert importer.read_case_package(tmp_path, tmp_path / "case_labels.json", custody=tmp_path / "custody.json")[0] == loaded
    assert _tree(tmp_path) == before
