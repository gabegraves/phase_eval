"""Scrub a string out of the content-addressed demo store and re-derive every hash that depends on it.

Usage: python tools/demo/rewrite_store.py <in.tar.xz> <out.tar.xz> <old> <new> <map.json>

Blobs are named by the SHA-256 of their bytes. Records are named by the SHA-256 of the canonical
JSON the workbench digests (sorted keys, compact separators, trailing newline). The script learns
which files name themselves and by which rule from the input, refuses to run if any rule or file
layout fails to reproduce the input bytes, rewrites to a fixed point from the original bytes each
round, and checks every rule again on the output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath

HEX64 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


LAYOUTS = {
    "indent4_sorted": lambda v: json.dumps(v, indent=4, sort_keys=True, allow_nan=False),
    "indent2_sorted": lambda v: json.dumps(v, indent=2, sort_keys=True, allow_nan=False),
    "indent2": lambda v: json.dumps(v, indent=2, allow_nan=False),
    "indent4": lambda v: json.dumps(v, indent=4, allow_nan=False),
    "compact_sorted": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False),
    "compact": lambda v: json.dumps(v, separators=(",", ":"), allow_nan=False),
    "default_sorted": lambda v: json.dumps(v, sort_keys=True, allow_nan=False),
    "default": lambda v: json.dumps(v, allow_nan=False),
    "compact_sorted_utf8": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False),
    "compact_utf8": lambda v: json.dumps(v, separators=(",", ":"), allow_nan=False, ensure_ascii=False),
    "indent2_sorted_utf8": lambda v: json.dumps(v, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False),
    "indent4_sorted_utf8": lambda v: json.dumps(v, indent=4, sort_keys=True, allow_nan=False, ensure_ascii=False),
}


def layout_of(text: str) -> tuple[str, str] | None:
    """The serializer and trailer that reproduce a JSON text exactly, or None."""
    body, trailer = (text[:-1], "\n") if text.endswith("\n") else (text, "")
    value = json.loads(body)
    for name, dump in LAYOUTS.items():
        if dump(value) == body:
            return name, trailer
    return None


def self_ids(path: str) -> list[str]:
    p = PurePosixPath(path)
    name = p.name[:-5] if p.name.endswith(".json") else p.name
    ids = HEX64.findall(name)
    if p.name == "attempt.json":
        ids += HEX64.findall(p.parent.name)
    return ids


def candidate_hashes(data: bytes) -> dict[str, str]:
    out = {"raw": sha(data)}
    try:
        out["canonical"] = sha(canonical(json.loads(data)))
    except (ValueError, UnicodeDecodeError):
        pass
    return out


def learn(files: dict[str, bytes]) -> dict[str, tuple[str, str]]:
    learned = {}
    for path, data in files.items():
        hashes = candidate_hashes(data)
        for ident in self_ids(path):
            rule = next((r for r, h in hashes.items() if h == ident), None)
            if rule:
                learned[path] = (ident, rule)
    return learned


def remap_str(text: str, mapping: dict[str, str]) -> str:
    return HEX64.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def sorted_like(original: list, current: list) -> list:
    if len(original) > 1 and all(isinstance(x, str) for x in original) and original == sorted(original):
        return sorted(current)
    if (
        len(original) > 1
        and all(isinstance(x, dict) and isinstance(x.get("sha256"), str) for x in original)
        and [x["sha256"] for x in original] == sorted(x["sha256"] for x in original)
    ):
        return sorted(current, key=lambda x: x["sha256"])
    return current


BLOB_SIZES: dict[str, int] = {}


def remap_json(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, dict):
        out = {remap_str(k, mapping): remap_json(v, mapping) for k, v in value.items()}
        if isinstance(out.get("sha256"), str) and out["sha256"] != value.get("sha256") and "size_bytes" in out:
            out["size_bytes"] = BLOB_SIZES[out["sha256"]]
        return out
    if isinstance(value, list):
        return sorted_like(value, [remap_json(v, mapping) for v in value])
    if isinstance(value, str):
        return remap_str(value, mapping)
    return value


def rewrite_text(text: str, mapping: dict[str, str]) -> str:
    """Remap one JSON document, keeping its exact layout; an unchanged value keeps its bytes."""
    value = json.loads(text)
    remapped = remap_json(value, mapping)
    if remapped == value:
        return text
    name, trailer = layout_of(text)
    body = LAYOUTS[name](remapped)
    return body + trailer


def rewrite_file(path: str, data: bytes, mapping: dict[str, str]) -> bytes:
    if path.startswith("store/blobs/") or not data.strip():
        return data
    text = data.decode()
    if path.endswith(".jsonl"):
        return "".join(
            rewrite_text(line, mapping) if line.strip() else line for line in text.splitlines(keepends=True)
        ).encode()
    try:
        json.loads(text)
    except ValueError:
        return remap_str(text, mapping).encode()
    return rewrite_text(text, mapping).encode()


def check_layouts(files: dict[str, bytes]) -> None:
    """Refuse to run unless every JSON document's layout is reproducible byte for byte."""
    bad = []
    for path, data in files.items():
        if path.startswith("store/blobs/") or not data.strip():
            continue
        text = data.decode()
        docs = [l for l in text.splitlines(keepends=True) if l.strip()] if path.endswith(".jsonl") else [text]
        for doc in docs:
            try:
                if layout_of(doc) is None:
                    bad.append(path)
                    break
            except ValueError:
                break
    if bad:
        sys.exit(f"layout not reproducible: {bad[:10]}")


def rewrite_store(src: str, dst: str, old: str, new: str, map_out: str) -> None:
    if not old or old == new:
        raise ValueError("old must be nonempty and differ from new")
    old_b, new_b = old.encode(), new.encode()
    if len({Path(path).resolve() for path in (src, dst, map_out)}) != 3:
        raise ValueError("input, output and hash-map paths must differ")
    BLOB_SIZES.clear()
    with tarfile.open(src, "r:xz") as tar:
        members = tar.getmembers()
        files = {m.name: tar.extractfile(m).read() for m in members if m.isfile()}
        dirs = [m.name for m in members if m.isdir()]

    for path, data in files.items():
        if path.startswith("store/blobs/") and sha(data) != PurePosixPath(path).name:
            sys.exit(f"input blob does not match its name: {path}")
    check_layouts(files)
    learned = learn(files)
    named = sorted(p for p in files if self_ids(p))
    by_rule: dict[str, int] = {}
    for _, rule in learned.values():
        by_rule[rule] = by_rule.get(rule, 0) + 1
    print(f"input: {len(files)} files, {len(dirs)} dirs; self-addressed: {by_rule}; names that are references: {len(named) - len(learned)}")

    for path, data in files.items():
        if path.startswith("store/blobs/"):
            BLOB_SIZES[sha(data.replace(old_b, new_b))] = len(data.replace(old_b, new_b))
    mapping: dict[str, str] = {}
    for round_no in range(1, 50):
        current = {}
        for path, data in files.items():
            scrubbed = data.replace(old_b, new_b)
            current[path] = scrubbed if path.startswith("store/blobs/") else rewrite_file(path, scrubbed, mapping)
        next_mapping = dict(mapping)
        for path, data in current.items():
            if data == files[path]:
                continue
            next_mapping[sha(files[path])] = sha(data)
            if path in learned:
                ident, rule = learned[path]
                next_mapping[ident] = candidate_hashes(data)[rule]
        next_mapping = {k: v for k, v in next_mapping.items() if k != v}
        if next_mapping == mapping:
            print(f"fixed point after {round_no} rounds; {len(mapping)} hashes remapped")
            break
        mapping = next_mapping
    else:
        sys.exit("no fixed point")

    out_files = {remap_str(p, mapping): d for p, d in current.items()}
    out_dirs = sorted({remap_str(d, mapping) for d in dirs})
    if len(out_files) != len(files):
        sys.exit("a rename collided")
    for path, data in out_files.items():
        if old_b in data:
            sys.exit(f"old string survives in {path}")
        if path.startswith("store/blobs/") and sha(data) != PurePosixPath(path).name:
            sys.exit(f"output blob does not match its name: {path}")
    relearned = learn(out_files)
    lost = {remap_str(p, mapping) for p in learned} - set(relearned)
    if lost:
        sys.exit(f"self-addressed files lost their rule: {sorted(lost)[:10]}")
    referenced = set()
    for path, data in out_files.items():
        referenced |= set(HEX64.findall(path))
        if not path.startswith("store/blobs/"):
            referenced |= set(HEX64.findall(data.decode(errors="ignore")))
    stale = referenced & set(mapping)
    if stale:
        sys.exit(f"old hashes still referenced: {sorted(stale)[:10]}")
    sizes = {PurePosixPath(p).name: len(d) for p, d in out_files.items() if p.startswith("store/blobs/")}
    wrong = []

    def check_sizes(v: object, where: str) -> None:
        if isinstance(v, dict):
            if isinstance(v.get("sha256"), str) and v["sha256"] in sizes and "size_bytes" in v and v["size_bytes"] != sizes[v["sha256"]]:
                wrong.append(where)
            for x in v.values():
                check_sizes(x, where)
        elif isinstance(v, list):
            for x in v:
                check_sizes(x, where)

    for path, data in out_files.items():
        if path.startswith("store/blobs/") or not data.strip():
            continue
        text = data.decode()
        for doc in (text.splitlines() if path.endswith(".jsonl") else [text]):
            if doc.strip():
                try:
                    check_sizes(json.loads(doc), path)
                except ValueError:
                    pass
    if wrong:
        sys.exit(f"size_bytes disagrees with the blob in: {sorted(set(wrong))[:10]}")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz", format=tarfile.PAX_FORMAT) as tar:
        for d in out_dirs:
            info = tarfile.TarInfo(d)
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, 0
            tar.addfile(info)
        for path in sorted(out_files):
            info = tarfile.TarInfo(path)
            info.size, info.mode, info.mtime = len(out_files[path]), 0o644, 0
            tar.addfile(info, io.BytesIO(out_files[path]))
    with open(dst, "xb") as fh:
        fh.write(buffer.getvalue())
    with open(map_out, "x") as fh:
        json.dump(mapping, fh, indent=1, sort_keys=True)
    changed = sorted(p for p in files if current[p] != files[p])
    print(f"output: {len(out_files)} files; content changed: {len(changed)}; renamed: {sum(1 for p in files if remap_str(p, mapping) != p)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("src", "dst", "old", "new", "map_out"):
        parser.add_argument(name)
    args = parser.parse_args(argv)
    rewrite_store(args.src, args.dst, args.old, args.new, args.map_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
