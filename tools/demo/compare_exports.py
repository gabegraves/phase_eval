"""Compare validated bundled datasets and CLI exports before and after a demo-store rewrite.

Usage: PYTHONPATH=src uv run --project launcher --with jsonschema python tools/demo/compare_exports.py <original.tar.xz> <rebuilt.tar.xz> <map.json> <old> <new>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path

from phaseeval.workbench import cli
from phaseeval.workbench.exports import dataset_json_schemas, validate_dataset

HEX64 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def export(tarball: Path, work: Path) -> dict[str, str]:
    with tarfile.open(tarball, "r:xz") as archive:
        archive.extractall(work / "demo", filter="data")
    for directory in (work / "demo/dataset", work / "export"):
        if directory.name == "export":
            result = cli.main(["export", "--out", str(directory), "--store", str(work / "demo/store")])
            if result:
                raise ValueError(f"CLI export failed for {tarball.name}")
        problems = validate_dataset(directory, dataset_json_schemas())
        if problems:
            raise ValueError(f"{tarball.name}: {problems}")
    return {f"{scope}/{p.name}": p.read_text()
            for scope, directory in (("dataset", work / "demo/dataset"), ("export", work / "export"))
            for p in sorted(directory.iterdir()) if p.is_file()}


UNORDERED_COLLECTIONS = {
    ("rights",),
    ("entries",),
    ("steps", "attempt", "inputs"),
    ("steps", "attempt", "outcome", "origin", "receipts"),
}


def canonical(value: object, path: tuple[str, ...] = ()) -> object:
    # These are re-derived from changed bytes or export time, rather than scientific facts.
    if isinstance(value, dict):
        return {key: canonical(item, (*path, key)) for key, item in value.items()
                if key not in ("size_bytes", "record_id", "judge_input_sha256", "frozen_at_utc")}
    if isinstance(value, list):
        items = [canonical(item, path) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True)) if path in UNORDERED_COLLECTIONS else items
    return value


def documents(name: str, text: str) -> object:
    if name.endswith(".jsonl"):
        # JSONL rows are a collection sorted by identities that the rewrite changes.
        return sorted((canonical(json.loads(line)) for line in text.splitlines() if line.strip()),
                      key=lambda item: json.dumps(item, sort_keys=True))
    return canonical(json.loads(text))


def compare_exports(original: Path, rebuilt: Path, mapping_path: Path, old: str, new: str) -> tuple[str, ...]:
    mapping = json.loads(mapping_path.read_text())
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        before, after = export(original, Path(a)), export(rebuilt, Path(b))
    if before.keys() != after.keys():
        raise ValueError("export file sets differ")
    bad = []
    for name in before:
        # The validator checks each manifest; its hashes and timestamps necessarily change.
        if Path(name).name == "manifest.json":
            continue
        mapped = HEX64.sub(lambda match: mapping.get(match.group(), match.group()), before[name]).replace(old, new)
        if documents(name, mapped) != documents(name, after[name]):
            bad.append(name)
    print(f"export files compared: {len(before) - 2}; differing beyond the scrub: {bad or 'none'}")
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
