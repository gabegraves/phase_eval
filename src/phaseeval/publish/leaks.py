from __future__ import annotations

import ipaddress
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, JsonValue


# ponytail: scanner DTOs are not stored or hashed. If persisted, adopt Record in P11 or later.
class LeakRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal[
        "ipv4", "ipv6", "posix_home", "posix_private", "windows_path", "local_host",
        "url_with_port", "email", "bearer_like", "custody_key", "run_identity", "demo_run",
    ]
    applies_to: Literal["keys", "values", "both"]


PUBLIC_RULES: tuple[LeakRule, ...] = tuple(
    LeakRule(name=name, applies_to="keys" if name == "custody_key" else "both")
    for name in get_args(LeakRule.model_fields["name"].annotation)
)

ALLOWED_PREFIXES: tuple[tuple[str, str], ...] = (
    ("/home/coder/svn-repositories/", "COD CIF provenance header (_cod_* lines), present in every COD candidate"),
    ("/private-runs/", "the redaction placeholder the original export wrote in place of host paths (U4)"),
)


class LeakFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rule: str
    pointer: str
    excerpt: str = Field(max_length=80)


_PATTERNS = {
    "ipv4": re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)"),
    "ipv6": re.compile(r"(?<!\w)(?=([0-9a-fA-F:]{0,39}:[0-9a-fA-F:.]{0,44}[0-9a-fA-F:](?:%[\w.-]+)?)(?![\w:-]))"),
    "posix_home": re.compile(r"/(?:Users|home|root)/[^\s\"'<>]*"),
    "posix_private": re.compile(r"/private/(?:var|tmp|etc)/[^\s\"'<>]*"),
    "windows_path": re.compile(r"(?<!\w)[A-Za-z]:[\\/][^\s\"<>]*|\\\\[\w.-]+\\[^\s\"<>]+"),
    "local_host": re.compile(r"\b(?:[\w-]+\.)+(?:local|ts\.net|internal)\b|\blocalhost(?::\d+)?\b", re.I),
    "url_with_port": re.compile(r"\b[a-z][a-z0-9+.-]*://(?:[^\s/?#@\"<>]*@)?(?:\[[^\]\s]+\]|[^\s/:@?#\"<>]+):\d+\b", re.I),
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,}\b"),
    "bearer_like": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}|\b(?:sk-|hf_|ghp_)[A-Za-z0-9_-]{16,}|\b[A-Za-z0-9_-]{32,}\b", re.I),
    "custody_key": re.compile(r"^(?:reference|reference_answer|reference_phase_coverage|truth|ground_truth|custody|withheld_context|split_check|canary|candidate_license_audit)$", re.I),
    "demo_run": re.compile(r"\bu6-(?:dev006|dev102)-qwen-(?:006|007|008)\b", re.I),
}


def _matches(text: str, rule: LeakRule, identity: Sequence[str]) -> bool:
    if rule.name == "run_identity":
        return any(name and name.casefold() in text.casefold() for name in identity)
    for match in _PATTERNS[rule.name].finditer(text):
        token = match.group(1) if rule.name == "ipv6" else match.group()
        if any(token.startswith(prefix) for prefix, _ in ALLOWED_PREFIXES):
            continue
        if rule.name in {"ipv4", "ipv6"}:
            try:
                address = ipaddress.ip_address(token)
            except ValueError:
                continue
            if address.version == 4 and not any(len(octet) > 1 for octet in token.split(".")):
                continue
        if (rule.name == "windows_path" and re.search(r"\b(?:as|in|if|while) $", text[max(0, match.start() - 32):match.start()])
                and re.fullmatch(r"[A-Za-z]:\\[nrt]", token)):
            # Python block headers in recorded code end with JSON-escaped newlines.
            continue
        if rule.name == "bearer_like" and not (re.match(r"Bearer\s+", token, re.I) or token.lower().startswith(("sk-", "hf_", "ghp_"))):
            # ponytail: raw tokens need mixed case, digits and entropy >=4.8;
            # add issuer patterns for other formats. Hex digests stay allowed.
            entropy = -sum((count / len(token)) * math.log2(count / len(token)) for count in Counter(token).values())
            if entropy < 4.8 or not (re.search(r"[a-z]", token) and re.search(r"[A-Z]", token) and re.search(r"\d", token)):
                continue
        return True
    return False


def _pointer(path: str, part: str | int) -> str:
    return path + "/" + str(part).replace("~", "~0").replace("/", "~1")


def scan(value: JsonValue, *, rules: Sequence[LeakRule] = PUBLIC_RULES, identity: Sequence[str] = ()) -> tuple[LeakFinding, ...]:
    """Inspect every JSON key and scalar, including list members."""
    findings = []

    def inspect(text: str, pointer: str, kind: str) -> None:
        for rule in rules:
            if rule.applies_to in (kind, "both") and _matches(text, rule, identity):
                findings.append(LeakFinding(rule=rule.name, pointer=pointer, excerpt=f"[redacted:{rule.name}]"))

    def visit(item: JsonValue, pointer: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                path = _pointer(pointer, key)
                inspect(key, path, "keys")
                visit(child, path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, _pointer(pointer, index))
        else:
            inspect(str(item), pointer, "values")

    visit(value, "")
    return tuple(findings)


def scan_bytes(data: bytes, *, rules: Sequence[LeakRule] = PUBLIC_RULES) -> tuple[LeakFinding, ...]:
    """Inspect JSON, JSONL, or unstructured artifact bytes without extracting a file."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # ponytail: binary artifacts expose ASCII runs of 8+ bytes; other encodings need a format reader.
        text = "\n".join(match.decode("ascii") for match in re.findall(rb"[\x20-\x7e\t\r\n]{8,}", data))
    try:
        return scan(json.loads(text), rules=rules)
    except ValueError:
        pass
    lines = [line for line in text.splitlines() if line.strip()]
    try:
        records = [json.loads(line) for line in lines]
    except ValueError:
        return scan(text, rules=rules)
    return scan(records, rules=rules)


def redact(value: JsonValue, findings: Sequence[LeakFinding]) -> JsonValue:
    """Withhold flagged fields and replace flagged list elements or root scalars."""
    pointers = {finding.pointer for finding in findings}

    def visit(item: JsonValue, pointer: str) -> JsonValue:
        if pointer in pointers:
            return "[redacted]"
        if isinstance(item, dict):
            return {key: visit(child, _pointer(pointer, key)) for key, child in item.items()
                    if _pointer(pointer, key) not in pointers}
        if isinstance(item, list):
            return [visit(child, _pointer(pointer, index)) for index, child in enumerate(item)]
        return item

    return visit(value, "")
