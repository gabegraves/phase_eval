"""Bounded native evidence pages; logical handles never become caller-supplied paths."""
from __future__ import annotations

import hashlib
from itertools import islice
import json
import math
from pathlib import Path

from phaseeval.workbench.evidence import Curve
from phaseeval.workbench.importers.external import parse_xy
from harness.physics.casedata import actor_context

STAGES = ("pawley", "rietveld_lattice", "rietveld_profile", "rietveld")


def checked_bytes(root: Path, name: str, digest: str | None = None) -> bytes:
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or any(p.startswith(".") for p in relative.parts):
        raise ValueError("evidence path is not an allowed relative path")
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("symlink evidence root is not allowed")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink evidence is not allowed")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("registered evidence is missing or outside its root")
    data = path.read_bytes()
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("registered evidence digest changed")
    return data


class Evidence:
    def __init__(self, inputs: Path, case_id: str, output: Path, trial: Path):
        self.inputs, self.output, self.trial = inputs.resolve(), output, trial
        manifest = checked_bytes(self.inputs, "manifest.json")
        self.manifest_sha256 = hashlib.sha256(manifest).hexdigest()
        matches = [c for c in json.loads(manifest)["cases"] if c["case_id"] == case_id]
        if len(matches) != 1:
            raise ValueError("case manifest must contain exactly one matching case")
        self.case = matches[0]
        library_name = f"candidates/{case_id}/manifest.json"
        library = checked_bytes(self.inputs, library_name)
        self.candidates = json.loads(library)["candidates"]
        self.files = {}
        self.register("scan", self.inputs, self.case["pattern_file"], "scan", self.case["pattern_sha256"], ".xy")
        for row in self.candidates:
            self.register("candidate:" + row["candidate_id"], self.inputs / "candidates" / case_id,
                          row["file"], "text", row["sha256"], ".cif")
        self.context = json.dumps(actor_context(self.case, self.manifest_sha256), ensure_ascii=False, allow_nan=False).encode()

    def register(self, handle, root, name, kind, digest=None, suffix=None):
        if suffix and Path(name).suffix.lower() != suffix:
            raise ValueError("registered evidence has the wrong file type")
        if digest is None:
            digest = hashlib.sha256(checked_bytes(root, name)).hexdigest()
        self.files[handle] = (root, name, kind, digest)

    def register_fit(self, label):
        name = f"{label}/result.json"
        try:
            result = checked_bytes(self.output, name)
        except (OSError, ValueError):
            return
        self.register(f"fit:{label}:result", self.output, name, "text", hashlib.sha256(result).hexdigest())
        stages = json.loads(result)
        for stage in STAGES:
            if isinstance(stages.get(stage), dict):
                name = f"{label}/{stage}.csv"
                try:
                    self.register(f"fit:{label}:{stage}", self.output, name, "curve")
                except (OSError, ValueError):
                    pass  # A saved stage without a curve is not readable curve evidence.

    def register_output(self, extra):
        self.register(extra["evidence_handle"], self.trial, extra["executor_output"], "text", extra["executor_sha256"])

    def read(self, *, handle=None, start=0, limit=None, two_theta_min=None, two_theta_max=None):
        if type(start) is not int or start < 0:
            raise ValueError("start must be a nonnegative integer")
        bounds = (two_theta_min, two_theta_max)
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in bounds):
            raise ValueError("angle bounds must be finite numbers")
        if all(v is not None for v in bounds) and two_theta_min > two_theta_max:
            raise ValueError("angle interval is reversed")
        if handle is None:
            kind, data = "text", json.dumps({"handles": ["context", *self.files], "row_limit": 100, "text_limit": 2000}).encode()
            handle = "catalog"
        elif handle == "context":
            checked_bytes(self.inputs, "manifest.json", self.manifest_sha256)
            kind, data = "text", self.context
        else:
            if not isinstance(handle, str) or handle not in self.files:
                raise ValueError("unknown evidence handle; read without a handle for the catalog")
            root, name, kind, digest = self.files[handle]
            data = checked_bytes(root, name, digest)
        maximum = 2000 if kind == "text" else 100
        limit = maximum if limit is None else limit
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise ValueError(f"limit must be an integer within 1..{maximum}")
        result = {"handle": handle, "kind": kind, "sha256": hashlib.sha256(data).hexdigest()}
        if kind == "text":
            if any(v is not None for v in bounds):
                raise ValueError("angle bounds apply only to numeric evidence")
            value = data.decode("utf-8")
            if start > len(value):
                raise ValueError("start exceeds text length")
            end = min(start + limit, len(value))
            return result | {"text": value[start:end], "total_chars": len(value), "offset_unit": "unicode_character",
                             "start": start, "end": end, "next_start": end if end < len(value) else None}
        if kind == "curve":
            curve = Curve.parse_csv(data)
            columns = ["two_theta", "observed", "calculated", "difference"] + (["background"] if curve.background is not None else [])
            rows = list(zip(*(getattr(curve, column) for column in columns)))
        else:
            lines = [line.split() for line in data.decode("utf-8").splitlines()
                     if line.strip() and not line.lstrip().startswith(("#", "!", "%"))]
            if any(len(row) != 2 for row in lines):
                raise ValueError("scan must contain exactly two columns")
            scan = parse_xy(data)
            if scan.ndim != 2 or scan.shape[1] != 2 or len(scan) == 0:
                raise ValueError("scan must contain two-column numeric rows")
            rows, columns = scan.tolist(), ["two_theta", "observed"]
            if any(not math.isfinite(v) for row in rows for v in row):
                raise ValueError("scan contains nonfinite values")
            if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
                raise ValueError("scan angles are not strictly increasing")
        if start > len(rows):
            raise ValueError("start exceeds native row count")
        indexes = list(islice((i for i in range(start, len(rows)) if
                   (two_theta_min is None or rows[i][0] >= two_theta_min) and
                   (two_theta_max is None or rows[i][0] <= two_theta_max)), limit + 1))
        page = indexes[:limit]
        return result | {"columns": columns, "units": ["degree_2theta"] + ["recorded_intensity"] * (len(columns) - 1),
                         "total_rows": len(rows), "row_indexes": page, "rows": [rows[i] for i in page],
                         "start": start, "end": page[-1] + 1 if page else start,
                         "angle_interval": list(bounds), "next_start": indexes[limit] if len(indexes) > limit else None}
