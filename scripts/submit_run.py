"""Submit a PhaseEval export to the public runs dataset as a reviewed pull request.

    python scripts/submit_run.py <dataset-dir>/episodes.jsonl [--dataset kineticdrive/phaseeval-runs] \\
        [--name <handle>] --i-hold-the-rights [--include-scans]

Submit episodes.jsonl from a validated dataset directory. Labels and bundled demo runs are
refused. Every scalar is scanned for private data before upload. The file is refused without
--i-hold-the-rights, or when scan/pattern bytes appear without --include-scans. A passing file
is uploaded with the caller's Hugging Face login as a pull request for review.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))

DEFAULT_DATASET = "kineticdrive/phaseeval-runs"
HF_ENDPOINT = "https://huggingface.co"
BYTES_KEYS = re.compile(r"(^|_)(bytes|b64|base64|content|data|raw)$")
BASE64_RUN = re.compile(r"^[A-Za-z0-9+/=\s]{4096,}$")
HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class Refused(Exception):
    pass


def walk(value: Any, path: str = "$") -> Iterator[tuple[str, str | None, Any]]:
    """(field path, key, value) for every key and leaf."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{path}.{key}", key, item
            yield from walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk(item, f"{path}[{i}]")
    else:
        yield path, None, value


def check(path: Path, include_scans: bool) -> list[dict[str, Any]]:
    """Parse episodes, check their scalars, then validate their dataset bundle."""
    from phaseeval.publish.leaks import PUBLIC_RULES, scan
    from phaseeval.workbench.exports import EpisodeRecord, check_submission
    from pydantic import ValidationError

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise Refused(f"{path.name}: not UTF-8") from exc
    records = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Refused(f"line {n}: not JSON ({exc.msg})") from exc
        try:
            EpisodeRecord.model_validate(record)
        except ValidationError as exc:
            raise Refused(f"line {n}: does not validate as episodes: {exc.errors()[0]['msg']}") from exc
        for field, key, value in walk(record, f"line {n}"):
            if not include_scans and (
                (BYTES_KEYS.search(key or "") and isinstance(value, str) and value)
                or (isinstance(value, str) and BASE64_RUN.match(value))
            ):
                raise Refused(f"{field}: carries scan/pattern bytes (pass --include-scans to submit them)")
        rules = tuple(r for r in PUBLIC_RULES if r.name != "demo_run")
        findings = scan(record, rules=rules) or scan(record)
        if findings:
            finding = findings[0]
            parts = finding.pointer.lstrip("/").split("/")
            field = f"line {n}" + "".join(f"[{part}]" if part.isdigit() else f".{part}" for part in parts)
            raise Refused(f"{field}: contains {finding.rule}")
        records.append(record)
    if not records:
        raise Refused(f"{path}: no records")
    if problems := check_submission([path]):
        raise Refused("; ".join(problems))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export", type=Path)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--name", default=None, help="submitter handle; defaults to your HF username")
    parser.add_argument("--i-hold-the-rights", action="store_true", dest="rights")
    parser.add_argument("--include-scans", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.rights:
            raise Refused("refused: pass --i-hold-the-rights to confirm you hold the rights to everything in the file")
        if args.name is not None and not HANDLE.match(args.name):
            raise Refused(f"refused: --name {args.name!r} is not a handle (letters, digits, . _ -)")
        if not args.export.is_file():
            raise Refused(f"{args.export}: no such file")
        records = check(args.export, args.include_scans)
    except Refused as exc:
        print(f"refused: {exc}".replace("refused: refused: ", "refused: "), file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(endpoint=HF_ENDPOINT)
    handle = args.name or api.whoami()["name"]
    digest = hashlib.sha256(args.export.read_bytes()).hexdigest()
    commit = api.upload_file(
        path_or_fileobj=str(args.export),
        path_in_repo=f"submissions/{handle}/{digest}.jsonl",
        repo_id=args.dataset,
        repo_type="dataset",
        create_pr=True,
        commit_message=f"Submit run {digest[:12]} from {handle}",
        commit_description=f"{len(records)} record(s), sha256 {digest}. Submitter confirms they hold the rights.",
    )
    print(commit.pr_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
