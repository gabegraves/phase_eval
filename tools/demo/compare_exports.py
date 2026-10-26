"""Compare validated bundled datasets and CLI exports before and after a demo-store rewrite.

Usage: PYTHONPATH=src uv run --locked python tools/demo/compare_exports.py <original.tar.xz> <rebuilt.tar.xz> <map.json> <old> <new>
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

from phaseeval.workbench import cli
from phaseeval.workbench.exports import dataset_json_schemas, validate_dataset
from rewrite_store import UNORDERED_COLLECTIONS, check_integrity, json_documents, remap_json


def export(tarball: Path, work: Path) -> tuple[dict[str, str], dict[str, int]]:
    with tarfile.open(tarball, "r:xz") as archive:
        archive.extractall(work / "demo", filter="data")
    sizes = {p.name: p.stat().st_size for p in (work / "demo/store/blobs").iterdir() if p.is_file()}
    for directory in (work / "demo/dataset", work / "export"):
        if directory.name == "export":
            result = cli.main(["export", "--out", str(directory), "--store", str(work / "demo/store")])
            if result:
                raise ValueError(f"CLI export failed for {tarball.name}")
        problems = validate_dataset(directory, dataset_json_schemas())
        if problems:
            raise ValueError(f"{tarball.name}: {problems}")
        for path in directory.iterdir():
            if path.is_file():
                for doc in json_documents(path.name, path.read_bytes()):
                    check_integrity(doc, sizes, f"{tarball.name}/{directory.name}/{path.name}")
    return {f"{scope}/{p.name}": p.read_text()
            for scope, directory in (("dataset", work / "demo/dataset"), ("export", work / "export"))
            for p in sorted(directory.iterdir()) if p.is_file()}, sizes


def canonical(value: object, path: tuple[str, ...] = ()) -> object:
    # These are re-derived from changed bytes or export time, rather than scientific facts.
    if isinstance(value, dict):
        return {key: canonical(item, (*path, key)) for key, item in value.items()
                if key not in ("judge_input_sha256", "frozen_at_utc")}
    if isinstance(value, list):
        items = [canonical(item, path) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True)) if path in UNORDERED_COLLECTIONS or path == ("snapshot_index",) else items
    return value


def documents(name: str, text: str, mapping: dict[str, str], sizes: dict[str, int]) -> object:
    if name.endswith(".jsonl"):
        # JSONL rows are a collection sorted by identities that the rewrite changes.
        return sorted((canonical(remap_json(json.loads(line), mapping, sizes)) for line in text.splitlines() if line.strip()),
                      key=lambda item: json.dumps(item, sort_keys=True))
    value = remap_json(json.loads(text), mapping, sizes)
    if Path(name).name == "manifest.json":
        # Export time and scrubbed bytes change these digests; record contents are compared separately.
        for key in ("generated_at_utc", "judge_manifest_sha256"):
            value.pop(key, None)
        value["episode_export"].pop("inputs_sha256", None)
        for entry in value["files"]:
            entry.pop("sha256", None)
    return canonical(value)


def compare_exports(original: Path, rebuilt: Path, mapping_path: Path, old: str, new: str) -> tuple[str, ...]:
    mapping = json.loads(mapping_path.read_text())
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        before, _ = export(original, Path(a))
        after, sizes = export(rebuilt, Path(b))
    if before.keys() != after.keys():
        raise ValueError("export file sets differ")
    bad = []
    for name in before:
        scrubbed = before[name].replace(old, new)
        if documents(name, scrubbed, mapping, sizes) != documents(name, after[name], {}, sizes):
            bad.append(name)
    print(f"export files compared: {len(before)}; differing beyond the scrub: {bad or 'none'}")
    return tuple(bad)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("original", "rebuilt", "mapping"):
        parser.add_argument(name, type=Path)
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args(argv)
    return int(bool(compare_exports(args.original, args.rebuilt, args.mapping, args.old, args.new)))


if __name__ == "__main__":
    sys.exit(main())
