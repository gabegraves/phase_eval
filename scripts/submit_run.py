"""Submit a PhaseEval export to the public runs dataset as a reviewed pull request.

    python scripts/submit_run.py <export.jsonl> [--dataset kineticdrive/phaseeval-runs] \\
        [--name <handle>] --i-hold-the-rights [--include-scans]

Every line must validate against one of the record schemas under schemas/ and carry no custody
field. The file is refused without --i-hold-the-rights, refused when a record carries scan or
pattern bytes unless --include-scans is given, and refused when any value holds an IPv4 address,
a /home/ or /Users/ path, or a ts.net host. A passing file is uploaded with the caller's own
Hugging Face login to submissions/<handle>/<sha256 of file>.jsonl as a pull request, and the PR
URL is printed. The only network traffic is huggingface_hub talking to huggingface.co.
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
LEAKS = {
    "an IPv4 address": re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    "a /home/ path": re.compile(r"/home/"),
    "a /Users/ path": re.compile(r"/Users/"),
    "a ts.net host": re.compile(r"ts\.net"),
}
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


def record_validators() -> list[Any]:
    from jsonschema import Draft202012Validator

    from phaseeval.workbench.exports import SCHEMA_FOR_FILE

    names = sorted(set(SCHEMA_FOR_FILE.values()))
    return [Draft202012Validator(json.loads((WORKTREE / "schemas" / n).read_text())) for n in names]


def check(path: Path, include_scans: bool) -> list[dict[str, Any]]:
    """The parsed records, or Refused naming the first problem."""
    from phaseeval.workbench.exports import CUSTODY_KEYS

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise Refused(f"{path}: no records")
    validators = record_validators()
    records = []
    for n, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Refused(f"line {n}: not JSON ({exc.msg})") from exc
        if not isinstance(record, dict) or not any(v.is_valid(record) for v in validators):
            raise Refused(f"line {n}: does not validate against any record schema in schemas/")
        for field, key, value in walk(record, f"line {n}"):
            if key in CUSTODY_KEYS:
                raise Refused(f"{field}: custody field")
            if not include_scans and (
                (BYTES_KEYS.search(key or "") and isinstance(value, str) and value)
                or (isinstance(value, str) and BASE64_RUN.match(value))
            ):
                raise Refused(f"{field}: carries scan/pattern bytes (pass --include-scans to submit them)")
            if isinstance(value, str):
                for what, pattern in LEAKS.items():
                    if pattern.search(value):
                        raise Refused(f"{field}: contains {what}")
            if key is not None:
                for what, pattern in LEAKS.items():
                    if pattern.search(key):
                        raise Refused(f"{field}: key contains {what}")
        records.append(record)
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
