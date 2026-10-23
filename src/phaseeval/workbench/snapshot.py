"""Frozen analysis snapshots and the workbench page's review payload (R12, R32, plan KTD4).

Target path: src/phaseeval/workbench/snapshot.py (public). Overnight contract commit (AXA-43zwk.8):
the record types and signatures are binding; the bodies are filled by the U4 view worker, which
ports the D v3 prototype's extract.py checks (K1, K2, K3, K5) into pure functions here.

- An AnalysisSnapshot is the exact evidence a blind rater is shown for one answer: pattern,
  fit, residual peaks, phase list and amounts, chemistry evidence, its originating attempt and
  its kind. It holds no reference answer, no run identity (model, harness, job, series) and no
  other judgment, so a rater and a model judge can be shown it before sign-off. Its digest keys
  every label.
- The review payload is the page's view of one snapshot. Its blind part derives from the
  snapshot and the actor-safe case view only; its revealed part (run identity, reference
  comparison, sibling episodes) is served by a separate route only after the store holds the
  rater's sign-off (labels.LabelStore reveal rules).

W3 view: an imported trial's receipts are the bytes of its Harbor directory without their
paths, so every file this module reads from them is identified by its own content (a .dia by
its TITEL line, a refine request and result by their working_dir, the trajectory by its shape)
and bound to the imported answer by value, never by a filename (R31). The answer file is the
one exception: the importer records its sha256 (ImportedSubmitted.answer_sha256), because a
trial's drafts can match the answer's phases and amounts.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
from pydantic import Field, JsonValue
from scipy.signal import find_peaks

from phaseeval.api.auth import Principal

from . import chemistry
from .chemistry import ChemistryEvidence
from .model import (
    AddPhases,
    ArtifactRef,
    AttemptRef,
    BlobRef,
    CaseManifest,
    Cell,
    ChildAttempt,
    Completed,
    Digest,
    FitSummary,
    ImportedRunResult,
    ImportedSubmitted,
    ImportedUnsuccessful,
    ImportRun,
    Incomplete,
    Record,
    Refine,
    RefineResult,
    RemovePhases,
    ReplacePhase,
    RootAttempt,
    SafeID,
)
from .triage import PROVISIONAL_2026_10_21 as POLICY
from .triage import Triage

SnapshotKind = Literal["agent_answer", "expert_correction", "reference"]


class SnapshotPhase(Record):
    phase_id: SafeID  # stable within the snapshot: the candidate id
    formula: str | None  # the candidate CIF's formula and space group (library metadata)
    space_group: str | None
    weight_percent: float | None
    weight_percent_error: float | None
    lattice_shift_percent: float | None = None
    # The phase as the answer described it, verbatim, kept apart from the CIF's metadata so a
    # wrong polymorph in the prose is not hidden (Science review, AXA-43zwk.10). None when the
    # answer named no such field, and for an expert correction.
    reported_formula: str | None = None
    reported_space_group: str | None = None
    reported_crystal_system: str | None = None
    reported_name: str | None = None


class ResidualPeak(Record):
    """One unexplained residual peak by the K2 rule (checkpoint-checks.md): residual smoothed
    over 0.08 deg, local maxima z >= 5, more than 1 deg from either scan end, unexplained when
    net observed > 5x net calculated; height as % of the strongest net observed peak."""

    peak_id: SafeID  # stable within the snapshot, e.g. "p23.95"
    two_theta_deg: float
    height_percent: float
    residual_counts: float  # the smoothed residual at the maximum (W3 view)


class SnapshotCheck(Record):
    code: Literal["K1", "K2", "K3", "K5"]
    name: str
    status: Literal["pass", "flag", "not_computable"]
    detail: str = Field(max_length=4000)
    policy_version: SafeID  # the triage.CheckPolicy version its thresholds came from


class AnalysisSnapshot(Record):
    schema_version: Literal["analysis-snapshot.v1"] = "analysis-snapshot.v1"
    kind: SnapshotKind
    origin: AttemptRef  # the attempt whose answer and fit are frozen (an ImportRun or a child)
    root: AttemptRef  # the pattern; reveal and assignment are per root
    case_id: SafeID
    pattern_sha256: BlobRef  # observed_pattern artifact
    fit_curve_sha256: BlobRef | None  # fit_curve artifact of the cited fit
    stage: str | None
    rwp_percent: float | None
    gof: float | None
    quantification: Literal["quantitative", "restricted_profile", "not_quantitative"] | None
    phases: tuple[SnapshotPhase, ...]
    residual_peaks: tuple[ResidualPeak, ...]
    checks: tuple[SnapshotCheck, ...]
    # The agent's own statements, shown after the evidence stage. Empty for a correction.
    claims: tuple[str, ...] = ()
    # What the answer said its amounts are (its quantity_basis, verbatim: text or an object).
    reported_quantity_basis: JsonValue = None
    # The answer file as submitted: every key, string, number, unit and null, wrong claims
    # included, apart from the native evidence above. Absolute paths and strings naming the run
    # become markers; fit_artifacts, a list of paths, is left out. None when the import bound no
    # answer file (FA worker, AXA-43zwk.21).
    reported_answer: JsonValue = None
    answered: bool  # False when the episode wrote no answer (the snapshot is its best fit)
    # Frozen with the rest so the digest covers all a rater or judge is shown (W3 view; approved
    # contract change).
    chemistry: ChemistryEvidence | None = None


MAX_POINTS = 1200  # curves are strided to at most this many points for the page
MAX_STEPS = 40
ESD_SOURCE = "BGMN final-stage .lst Q value esd of the cited fit, times 100 (wt%)"
SHIFT_SOURCE = "max |a, b, c| shift of the cited fit's final stage from the candidate CIF cell"
# Every statement in the agent's analysis.json is a claim, kind named by its top-level key,
# except these: ids and amounts, the quantity basis (frozen on its own), artifact paths and fit
# statistics, and inside any value its boilerplate definition or unit label. Of the phases, only
# their evidence text.
NOT_CLAIMS = frozenset(
    {"case_id", "weight_fractions", "quantity_basis", "fit_artifacts", "fit_statistics"}
)
CLAIM_KINDS = {
    "phases": "evidence",
    "uncertainties": "uncertainty",
    "limitations": "limitation",
    "fit_performed": "method",
}
MIN_CLAIM_WORDS = 3  # a statement, not a unit or a label
# An absolute path of two or more segments that starts a token, e.g. /workspace/output/a.json.
# The lookbehind keeps "P2_1/c", "Rwp/Rexp" and the slashes of a URL.
ABSOLUTE_PATH = re.compile(r"(?<![\w.)\]}/])/(?:[\w.+-]+/)+[\w.+-]*")
PATH_MARK = "[path]"
IDENTITY_MARK = "[run identity]"
NOT_SUBMITTED = frozenset({"fit_artifacts"})  # paths only, by the U6 response contract
REPORTED_FIELDS = {
    "formula": "reported_formula",
    "space_group": "reported_space_group",
    "crystal_system": "reported_crystal_system",
    "phase_name": "reported_name",
}
CURVE_PRECISION = "4 significant digits"
RESIDUAL_WINDOW_DEG = 0.5  # native rows either side of a listed residual peak
# A package withholds precursor amounts with a notice that can carry the packager's rationale
# ("the amounts approximate the reference weight fractions"): a policy, not measured
# composition, so the brief says only that they are withheld (Science, AXA-43zwk.8 09:54 UTC).
WITHHELD_AMOUNTS = "withheld by task policy"
PHASE_WINDOWS = 5  # native windows per reported phase, its strongest calculated maxima first
PHASE_WINDOW_RULE = (
    f"For each reported phase, up to {PHASE_WINDOWS} windows of every native row within"
    f" {RESIDUAL_WINDOW_DEG:g} deg of a local maximum of that phase's calculated contribution in the"
    f" fit's .dia, strongest first; a maximum within {2 * RESIDUAL_WINDOW_DEG:g} deg of a"
    " stronger chosen one is skipped, so one phase's windows never overlap. Chosen from the saved"
    " fit alone, by the same rule for every answer. A finite selection of calculated peaks:"
    " supporting evidence at those positions, not a complete test of the phase; the number of"
    " windows says nothing about detection limits or absent phases."
)


class SnapshotError(ValueError):
    """The attempt is not one a snapshot can freeze, or it carries custody bytes."""


# --- formulas, CIFs ----------------------------------------------------------------------------


def short_formula(formula_sum: str | None) -> str | None:
    """'C Li2 O3' -> 'Li2CO3': other elements in listed order, then C, H, O."""
    if not formula_sum:
        return None
    tokens = formula_sum.split()
    tail = {"C": 0, "H": 1, "O": 2}

    def element(token: str) -> str:
        match = re.match(r"[A-Z][a-z]?", token)
        return match.group(0) if match else token

    head = [t for t in tokens if element(t) not in tail]
    rest = sorted((t for t in tokens if element(t) in tail), key=lambda t: tail[element(t)])
    return "".join(head + rest)


def hm_short(hm: str | None) -> str | None:
    """'P 1 21/c 1' -> 'P2_1/c'; 'F m -3 m' -> 'Fm-3m'; 'P n a 21' -> 'Pna2_1'."""
    if not hm:
        return None
    parts = hm.split()
    if len(parts) == 4 and parts[1] == "1" and parts[3] == "1":
        parts = [parts[0], parts[2]]
    return re.sub(r"(\d)(\d)", r"\1_\2", "".join(parts))


@dataclass(frozen=True)
class CifInfo:
    space_group: str | None  # e.g. "Pna2_1 (33)"
    cell: Mapping[str, float | None]


def cif_info(text: str) -> CifInfo:
    def get(key: str) -> str | None:
        match = re.search(rf"^{key}\s+(.+?)\s*$", text, re.M)
        return match.group(1).strip().strip("'\"") if match else None

    def number(key: str) -> float | None:
        value = get(key)
        try:
            return float(re.sub(r"\(.*\)", "", value)) if value else None
        except ValueError:
            return None

    hm = get("_symmetry_space_group_name_H-M") or get("_space_group_name_H-M_alt")
    raw = get("_space_group_IT_number") or get("_symmetry_Int_Tables_number")
    number_it = int(raw) if raw and raw.isdigit() else None
    space_group = f"{hm_short(hm)} ({number_it})" if hm else None
    return CifInfo(space_group, {axis: number(f"_cell_length_{axis}") for axis in "abc"})


@dataclass(frozen=True)
class Library:
    """The actor-safe candidate library of one case: formula and CIF facts by candidate id."""

    formula: Mapping[str, str | None]
    cif: Mapping[str, CifInfo]

    def space_group(self, candidate_id: str) -> str | None:
        info = self.cif.get(candidate_id)
        return info.space_group if info else None


def _library(store: Any, case: CaseManifest) -> Library:
    return Library(
        formula={c.candidate_id: short_formula(c.formula_sum) for c in case.candidates},
        cif={
            c.candidate_id: cif_info(store.read_blob(c.cif.sha256).decode(errors="replace"))
            for c in case.candidates
        },
    )


# --- BGMN files ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dia:
    """A BGMN .dia: 2theta, observed, calculated, background, then one column per phase."""

    label: str | None  # the refine run label from the TITEL path, when it names one
    stage: str | None
    phase_ids: tuple[str, ...]
    rows: np.ndarray


TITEL = re.compile(rb"^TITEL=(\S+)")


def read_dia(data: bytes) -> Dia:
    header, _, body = data.partition(b"\n")
    title = TITEL.match(header)
    label = stage = None
    if title:
        path = PurePosixPath(title.group(1).decode(errors="replace"))
        label = path.parent.parent.name if path.parent.name == "fit" else None
        stage = path.name.split("_", 1)[1] if "_" in path.name else None
    ids = tuple(re.findall(r"STRUC\[\d+\]=(\S+)", header.decode(errors="replace")))
    width = 4 + len(ids)
    rows = [
        [float(x) for x in parts[:width]]
        for parts in (line.split() for line in body.decode().splitlines())
        if len(parts) >= width
    ]
    return Dia(label, stage, ids, np.array(rows, dtype=float).reshape(-1, width))


def read_curve(data: bytes) -> np.ndarray:
    """A fit_curve csv: two_theta, observed, calculated, difference."""
    return np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, ndmin=2)


def _sig4(x: float) -> float | None:
    x = float(x)
    if not math.isfinite(x):
        return None
    return x if x == 0 else float(f"{x:.4g}")


def _r(x: float | None, digits: int = 2) -> float | None:
    return None if x is None else round(float(x), digits)


# --- checkpoint checks (docs/research/2026-10-21-eval-direction/checkpoint-checks.md) -----------


@dataclass(frozen=True)
class FitRecord:
    """One refine run's final-stage facts, from a recorded result or a live RefineResult."""

    label: str
    candidates: tuple[str, ...]
    failed: bool
    stages: Mapping[str, Mapping[str, Any]]  # stage -> {rwp, gof, converged}
    final: bool
    rwp: float | None
    gof: float | None
    phases: tuple[Mapping[str, Any], ...]  # {id, wt, esd, cell}
    stage_degradation: Mapping[str, Any] | None = None
    elapsed_seconds: float | None = None


def k1(fit: FitRecord) -> dict[str, Any]:
    degradation = fit.stage_degradation
    if degradation:
        flagged = bool(degradation.get("rietveld_degraded_after_good_pawley"))
        pawley, final = degradation.get("pawley_rwp"), degradation.get("rietveld_rwp")
        source = "the run's result.json stage_degradation (written by refine.py)"
    else:
        p, r = fit.stages.get("pawley"), fit.stages.get("rietveld")
        pawley, final = (p or {}).get("rwp"), (r or {}).get("rwp")
        usable = bool(
            not fit.failed
            and p
            and r
            and p.get("converged")
            and r.get("converged")
            and pawley is not None
            and final is not None
        )
        flagged = bool(usable and pawley <= 15.0 and final - pawley >= 10.0)
        source = "recomputed with the coded rule (src/phaseeval/refinement/protocol.py)"
    if pawley is None or final is None:
        return _check(
            "K1",
            "Rietveld degraded after a good Pawley fit",
            "not_computable",
            "Stage Rwp missing.",
        )
    detail = (
        f"rietveld_degraded_after_good_pawley = {str(flagged).lower()}. Pawley Rwp {pawley}%, final "
        f"Rietveld Rwp {final}%, difference {final - pawley:+.2f} points. Rule: Pawley Rwp <= 15 and "
        f"Rietveld minus Pawley >= 10. Source: {source}."
    )
    return _check(
        "K1", "Rietveld degraded after a good Pawley fit", "flag" if flagged else "pass", detail
    )


def k2(dia: Dia) -> dict[str, Any]:
    """Unexplained residual peaks of the final-stage .dia; the extract.py rule, verbatim."""
    policy = POLICY
    tt, yo, yc, yb = (dia.rows[:, i] for i in range(4))
    step = float(np.median(np.diff(tt)))
    n = max(3, int(round(0.08 / step)) | 1)
    kernel = np.ones(n) / n
    d = np.convolve(yo - yc, kernel, "same")
    so, sc, sb = (np.convolve(a, kernel, "same") for a in (yo, yc, yb))
    sigma = np.sqrt(np.convolve(np.maximum(yo, 1.0), kernel, "same") / n)
    z = d / sigma
    net_o, net_c = so - sb, sc - sb
    strongest = float(np.max(net_o))
    lo, hi = tt[0] + 1.0, tt[-1] - 1.0
    maxima = []
    for i in find_peaks(d)[0]:
        if z[i] < 5 or not lo <= tt[i] <= hi:
            continue
        ratio = math.inf if net_c[i] <= 0 else net_o[i] / net_c[i]
        if ratio <= 5:
            continue
        maxima.append(
            {
                "two_theta": round(float(tt[i]), 3),
                "pct_of_strongest_peak": round(100 * float(d[i]) / strongest, 2),
                "residual_counts": round(float(d[i]), 1),
                "z": round(float(z[i]), 1),
                "net_obs_over_net_calc": None if math.isinf(ratio) else round(float(ratio), 2),
            }
        )
    # adjacent maxima of one residual feature (consecutive gaps <= 0.1 deg) are reported once
    clusters: list[list[dict[str, Any]]] = []
    for peak in sorted(maxima, key=lambda m: m["two_theta"]):
        if clusters and peak["two_theta"] - clusters[-1][-1]["two_theta"] <= 0.1:
            clusters[-1].append(peak)
        else:
            clusters.append([peak])
    listed, below = [], 0
    for cluster in clusters:
        top = dict(max(cluster, key=lambda m: m["pct_of_strongest_peak"]))
        top["merged_maxima"] = len(cluster)
        if top["pct_of_strongest_peak"] < policy.k2_list_height_percent:
            below += 1
            continue
        top["status"] = (
            "flag" if top["pct_of_strongest_peak"] >= policy.k2_flag_height_percent else "weak"
        )
        top["peak_id"] = f"p{top['two_theta']:.2f}"
        listed.append(top)
    listed.sort(key=lambda c: -c["pct_of_strongest_peak"])
    highest = listed[0]["pct_of_strongest_peak"] if listed else 0.0
    low, high = f"{policy.k2_list_height_percent:g}%", f"{policy.k2_flag_height_percent:g}%"
    if listed:
        found = ", ".join(f"{p['pct_of_strongest_peak']}% at {p['two_theta']} deg" for p in listed)
        detail = f"Unexplained residual peaks at or above {low} of the strongest net observed peak: {found}."
    else:
        detail = (
            f"No unexplained residual peak at or above {low} of the strongest net observed peak."
        )
    if below:
        detail += f" {below} more unexplained residual features sit below {low}."
    detail += (
        " Method: residual smoothed over 0.08 deg, local maxima with z >= 5 more than 1 deg from"
        " either scan end, unexplained when net observed > 5x net calculated, height as % of the"
        " strongest net observed peak; adjacent maxima within 0.1 deg are reported once at the"
        f" highest. Flag at >= {high}; {low} to {high} is a weak unassigned peak, shown without a"
        f" flag. Policy {policy.version}."
    )
    status = "flag" if highest >= policy.k2_flag_height_percent else "pass"
    return _check("K2", "Unexplained residual peaks", status, detail) | {
        "max_pct": highest,
        "peaks": listed,
    }


def lattice_rows(fit: FitRecord, library: Library) -> list[dict[str, Any]]:
    rows = []
    for phase in fit.phases:
        initial = library.cif[phase["id"]].cell if phase["id"] in library.cif else {}
        shifts = {}
        for axis in "abc":
            value, start = (phase.get("cell") or {}).get(axis), initial.get(axis)
            if value and start:
                shifts[axis] = 100.0 * (value - start) / start
        rows.append(
            {
                "candidate_id": phase["id"],
                "wt_pct": _r(phase.get("wt")),
                "max_abs_axis_shift_pct": round(max(abs(x) for x in shifts.values()), 3)
                if shifts
                else None,
                "shifts_pct": {k: round(v, 3) for k, v in shifts.items()},
            }
        )
    return rows


def k3(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    big = [p for p in rows if (p["wt_pct"] or 0) >= 2]
    hits = [p for p in big if (p["max_abs_axis_shift_pct"] or 0) >= 1.0]
    worst = max(big, key=lambda p: p["max_abs_axis_shift_pct"] or 0) if big else None
    if hits:
        detail = "Cell far from its CIF: " + ", ".join(
            f"{p['candidate_id']} ({p['wt_pct']} wt%) max axis shift {p['max_abs_axis_shift_pct']}%"
            for p in hits
        )
        detail += "."
    else:
        detail = (
            "No phase of at least 2 wt% has an a, b or c axis shifted 1% or more from its CIF cell."
        )
        if worst:
            detail += f" Largest: {worst['candidate_id']} {worst['max_abs_axis_shift_pct']}%."
    detail += " Advisory: names a cell far from its CIF, not a wrong phase."
    status = "flag" if hits else "pass"
    return _check("K3", "Lattice shift from the CIF cell", status, detail) | {
        "advisory": True,
        "phases": list(rows),
    }


def k5(answer_percent: Mapping[str, float | None] | None, fit: FitRecord) -> dict[str, Any]:
    name = "Answer equals the cited fit"
    if answer_percent is None:
        return _check("K5", name, "not_computable", "No answer cites this fit.")
    if not fit.final:
        return _check("K5", name, "flag", "The cited fit has no Rietveld stage.")
    fit_wt = {p["id"]: p["wt"] for p in fit.phases}
    diffs = {
        cid: None if wt is None or fit_wt.get(cid) is None else round(wt - fit_wt[cid], 4)
        for cid, wt in answer_percent.items()
    }
    extra = sorted(set(fit_wt) - set(answer_percent))
    tolerance = POLICY.k5_flag_diff_wt_pct
    ok = all(v is not None and abs(v) <= tolerance for v in diffs.values()) and not extra
    detail = (
        f"Answer fractions against the final Rietveld stage of {fit.label}: differences in wt% points "
        + ", ".join(
            f"{k} {v:+.4f}" if v is not None else f"{k} not in fit" for k, v in diffs.items()
        )
        + (f"; fit phases missing from the answer: {extra}" if extra else "")
        + f". Tolerance {tolerance} points (answer fractions are rounded to 4 decimals). The cited stage is Rietveld, not Pawley."
    )
    return _check("K5", name, "pass" if ok else "flag", detail)


def _check(code: str, name: str, status: str, detail: str) -> dict[str, Any]:
    return {
        "code": code,
        "name": name,
        "status": status,
        "detail": detail,
        "policy_version": POLICY.version,
    }


# --- an imported trial's receipts ---------------------------------------------------------------


@dataclass
class Receipts:
    """The files of one imported trial, identified by content. Nothing here is a path."""

    dias: dict[tuple[str, str], bytes] = field(default_factory=dict)  # (label, stage) -> .dia
    curves: list[bytes] = field(default_factory=list)  # fit_curve csvs
    requests: dict[str, Mapping[str, Any]] = field(default_factory=dict)  # label -> request.json
    results: dict[str, Mapping[str, Any]] = field(default_factory=dict)  # label -> result.json
    trajectory: Mapping[str, Any] | None = None
    rollout: bytes | None = None
    # .dia files whose TITEL names no run label or stage: a saved BGMN or Profex project's
    unlabeled: list[bytes] = field(default_factory=list)


def _label_of(working_dir: object) -> str | None:
    if not isinstance(working_dir, str):
        return None
    path = PurePosixPath(working_dir)
    return (path.parent if path.name == "fit" else path).name or None


def read_receipts(store: Any, refs: Sequence[ArtifactRef]) -> Receipts:
    found = Receipts()
    for ref in refs:
        data = store.read_blob(ref.sha256)
        if data.startswith(b"TITEL="):
            dia = read_dia(data)
            if dia.label and dia.stage:
                found.dias[(dia.label, dia.stage)] = data
            else:
                found.unlabeled.append(data)
        elif data.startswith(b"two_theta,observed,calculated,difference"):
            found.curves.append(data)
        elif data.startswith(b'{"timestamp"') and b'"session_meta"' in data[:4096]:
            found.rollout = data
        elif data.lstrip()[:1] == b"{":
            try:
                value = json.loads(data)
            except ValueError:
                continue
            if not isinstance(value, dict):
                continue
            config = value.get("config")
            if "candidates" in value and "pattern_sha256" in value and isinstance(config, dict):
                if label := _label_of(config.get("working_dir")):
                    found.requests[label] = value
            elif "working_dir" in value and "refinement_failed" in value and "pawley" in value:
                if label := _label_of(value.get("working_dir")):
                    found.results[label] = value
            elif "steps" in value and "session_id" in value and "schema_version" in value:
                found.trajectory = value
    return found


def _recorded_fit(
    label: str, result: Mapping[str, Any], request: Mapping[str, Any] | None
) -> FitRecord:
    final = result.get("rietveld")
    stages = {
        stage: {"rwp": s.get("rwp"), "gof": s.get("gof"), "converged": s.get("converged")}
        for stage in ("pawley", "rietveld_lattice", "rietveld_profile", "rietveld")
        if (s := result.get(stage))
    }
    phases = tuple(
        {
            "id": p["phase_id"],
            "wt": p.get("weight_fraction"),
            "esd": None if p.get("weight_fraction_error") is None else p["weight_fraction_error"],
            "cell": p.get("lattice_params"),
        }
        for p in (final or {}).get("phases", ())
    )
    return FitRecord(
        label=label,
        candidates=tuple(c["candidate_id"] for c in (request or {}).get("candidates", ())),
        failed=bool(result.get("refinement_failed")),
        stages=stages,
        final=final is not None,
        rwp=final.get("rwp") if final else None,
        gof=final.get("gof") if final else None,
        phases=phases,
        stage_degradation=result.get("stage_degradation"),
        elapsed_seconds=result.get("elapsed_seconds"),
    )


def _live_fit(label: str, result: RefineResult) -> FitRecord:
    final = result.final
    stages = {
        s.stage: {"rwp": s.rwp_percent, "gof": s.gof, "converged": s.converged}
        for s in result.stages
    }
    phases = tuple(
        {
            "id": p.candidate_id,
            "wt": p.weight_percent,
            "esd": p.weight_percent_error,
            "cell": _cell(p.cell),
        }
        for p in (final.phases if final else ())
    )
    return FitRecord(
        label=label,
        candidates=tuple(p["id"] for p in phases),
        failed=result.refinement_failed,
        stages=stages,
        final=final is not None and final.stage == "rietveld",
        rwp=final.rwp_percent if final else None,
        gof=final.gof if final else None,
        phases=phases,
    )


def _cell(cell: Cell) -> dict[str, float | None]:
    return {"a": cell.a, "b": cell.b, "c": cell.c}


def _matches(fit: FitRecord, summary: FitSummary) -> bool:
    """A recorded result is the fit summary's run when its final stage carries the same values."""
    if (
        not fit.final
        or fit.rwp is None
        or not math.isclose(fit.rwp, summary.rwp_percent, abs_tol=1e-9)
    ):
        return False
    weights = {p["id"]: p["wt"] for p in fit.phases}
    return set(weights) == {p.candidate_id for p in summary.phases} and all(
        p.weight_percent is None
        or (
            weights[p.candidate_id] is not None
            and math.isclose(weights[p.candidate_id], p.weight_percent, abs_tol=1e-6)
        )
        for p in summary.phases
    )


def _first_calculated(data: bytes, separator: bytes | None) -> float | None:
    """The calculated value of the first data row: a cheap test before a full parse."""
    lines = data.split(b"\n", 2)
    try:
        return float(lines[1].split(separator)[2])
    except (IndexError, ValueError):
        return None


def _dia_for_curve(dias: Mapping[tuple[str, str], bytes], curve: np.ndarray) -> Dia | None:
    """The rietveld-stage .dia whose observed and calculated columns are the curve's."""
    for (_, stage), data in sorted(dias.items()):
        if stage == "rietveld" and (dia := _dia_matching(data, curve)) is not None:
            return dia
    return None


def _dia_matching(data: bytes, curve: np.ndarray) -> Dia | None:
    """The .dia when its observed and calculated columns are the curve's, row for row."""
    first = _first_calculated(data, None)
    if first is None or not math.isclose(first, curve[0, 2], abs_tol=1e-6):
        return None
    dia = read_dia(data)
    if dia.rows.shape[0] == curve.shape[0] and np.allclose(dia.rows[:, 1:3], curve[:, 1:3], atol=1e-6):
        return dia
    return None


def _named_by_candidates(dia: Dia, candidates: Sequence[str]) -> Dia:
    """The .dia's phase columns under the candidate ids they stand for. BGMN names a phase by the
    processed form of its candidate id (str_generator._process_phase_name, how the harness and
    the executors name a candidate's .str), so a column is renamed only when every column maps
    to exactly one candidate that way; otherwise the names stay as BGMN wrote them."""
    from phaseeval.refinement.bgmn.str_generator import _process_phase_name

    names: dict[str, list[str]] = {}
    for cid in candidates:
        names.setdefault(_process_phase_name(cid), []).append(cid)
    if set(dia.phase_ids) == set(candidates) or not dia.phase_ids:
        return dia
    if all(len(names.get(name, ())) == 1 for name in dia.phase_ids):
        return replace(dia, phase_ids=tuple(names[name][0] for name in dia.phase_ids))
    return dia


IMPORTED_FIT_LABEL = "imported refinement"


def _summary_fit(summary: FitSummary) -> FitRecord:
    """The fit of an import whose receipts hold no recorded refine run: a refinement finished in
    other software (importers.external). Its summary is the fit, as the importer read it."""
    phases = tuple(
        {"id": p.candidate_id, "wt": p.weight_percent, "esd": p.weight_percent_error, "cell": _cell(p.cell)}
        for p in summary.phases
    )
    return FitRecord(
        label=IMPORTED_FIT_LABEL,
        candidates=tuple(p["id"] for p in phases),
        failed=False,
        stages={summary.stage: {"rwp": summary.rwp_percent, "gof": summary.gof, "converged": None}},
        final=summary.stage == "rietveld",
        rwp=summary.rwp_percent,
        gof=summary.gof,
        phases=phases,
    )


def _curve_for_dia(curves: Sequence[bytes], dia: Dia) -> bytes | None:
    for data in curves:
        first = _first_calculated(data, b",")
        if first is None or not math.isclose(first, dia.rows[0, 2], abs_tol=1e-6):
            continue
        curve = read_curve(data)
        if curve.shape[0] == dia.rows.shape[0] and np.allclose(
            dia.rows[:, 1:3], curve[:, 1:3], atol=1e-6
        ):
            return data
    return None


# --- trajectory -> steps (extract.py build_steps) ----------------------------------------------

INPUT_READS = {"the case brief", "the candidate table", "the case manifest", "the tool guide"}
PHRASES = [
    (lambda c: "case_brief.py" in c, ("Read", "the case brief")),
    (
        lambda c: "candidates.py" in c or re.search(r"candidates/DEV\d+/manifest\.json", c),
        ("Read", "the candidate table"),
    ),
    (lambda c: "screen.py" in c and "--residual" in c, ("Screened", "residual peaks")),
    (lambda c: "screen.py" in c, ("Screened", "candidate phases")),
    (lambda c: "/input/manifest.json" in c, ("Read", "the case manifest")),
    (lambda c: "README" in c, ("Read", "the tool guide")),
    (lambda c: "analysis.json" in c, ("Re-read", "the draft answer")),
    (lambda c: "request.json" in c, ("Read", "refinement requests")),
    (
        lambda c: re.search(r"d_to_2th|hkl|arcsin|2theta|d-spacing|d_spacing", c),
        ("Computed", "reflection positions"),
    ),
    (lambda c: re.search(r"\.cif\b|\.str\b", c), ("Read", "CIF structures")),
    (lambda c: re.search(r"result\.json|\.lst\b", c), ("Read", "fit results")),
    (
        lambda c: re.search(r"rietveld[_a-z]*\.csv|pawley\.csv|\.dia\b", c),
        ("Inspected", "the difference curve"),
    ),
    (lambda c: re.search(r"matplotlib|\.png", c), ("Plotted", "the pattern")),
    (lambda c: re.search(r"\.xy\b", c), ("Inspected", "the raw pattern")),
    (
        lambda c: re.search(r"import (cif|pymatgen)|from pymatgen|dir\(|pip |help\(", c),
        ("Probed", "installed packages"),
    ),
    (
        lambda c: re.search(r"\bls\b|\bfind\b|\bwc\b|^\s*(head|tail|du)\b", c),
        ("Listed", "output files"),
    ),
    (lambda c: re.search(r"^\s*mkdir", c), ("Created", "the output folder")),
    (lambda c: "python" in c, ("Ran", "numerical probes")),
]
ANSWER_WRITE = re.compile(
    r"(cat|tee)\s*>+\s*\S*analysis\.json|>\s*/workspace/output/analysis\.json"
    r"|open\(\s*['\"][^'\"]*analysis\.json['\"]\s*,\s*['\"]w"
    r"|analysis_path[^\n]*\n[^\n]*open\([^)]*['\"]w"
)
STEP_ORDER = (
    "kind", "title", "run_label", "phases", "t", "t_start", "t_end", "rwp", "gof", "finished_rietveld",
    "calls", "from_script", "compactions_inside", "attempt", "call_index", "step_kind",
)  # fmt: skip
STEP_KIND = {"segment": "segment", "refine": "refine", "answer": "tool_call"}


def _phrase(cmd: str) -> tuple[str, str]:
    return next((ph for test, ph in PHRASES if test(cmd)), ("Ran", "shell commands"))


def _title(phrases: list[tuple[str, str]], first: bool) -> str:
    """Plain-words title: up to 3 verb groups (by call count, first-seen order), 2 objects each."""
    if not first and len({o for v, o in phrases if v == "Read" and o in INPUT_READS}) >= 2:
        phrases = [
            ("Re-read", "the case inputs") if v == "Read" and o in INPUT_READS else (v, o)
            for v, o in phrases
        ]
    count: dict[tuple[str, str], int] = {}
    seen: list[tuple[str, str]] = []
    for ph in phrases:
        count[ph] = count.get(ph, 0) + 1
        if ph not in seen:
            seen.append(ph)
    verbs: list[str] = []
    for verb, _ in seen:
        if verb not in verbs:
            verbs.append(verb)
    per_verb = {v: sum(n for (vv, _), n in count.items() if vv == v) for v in verbs}
    keep = sorted(sorted(verbs, key=lambda v: -per_verb[v])[:3], key=verbs.index)
    parts = []
    for verb in keep:
        objects = [o for vv, o in seen if vv == verb]
        objects = sorted(sorted(objects, key=lambda o: -count[(verb, o)])[:2], key=objects.index)
        parts.append(f"{verb.lower()} " + " and ".join(objects))
    title = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", then " + parts[-1]
    return title[0].upper() + title[1:]


def _is_answer_write(cmd: str) -> bool:
    if "analysis.json" not in cmd:
        return False
    if ANSWER_WRITE.search(cmd):
        return True
    match = re.search(r"(\w+)\s*=\s*['\"][^'\"]*analysis\.json['\"]", cmd)
    return bool(match and re.search(rf"open\(\s*{match.group(1)}\s*,\s*['\"]w", cmd))


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_steps(
    trajectory: Mapping[str, Any],
    recorded: Sequence[Any],
    fits: Mapping[str, FitRecord],
    library: Library,
    attempt: str,
    rollout: bytes | None,
) -> list[dict[str, Any]]:
    """The step list and refinement strip of an imported trajectory. Calls are enumerated as the
    importer enumerated them, and each must hash to its RecordedToolCall row, so every step's
    call_index is the index the store recorded. Times are seconds from the first trajectory step
    (the receipts carry no file times)."""
    steps_in = trajectory["steps"]
    t0 = _time(next(s["timestamp"] for s in steps_in if s.get("timestamp")))
    calls = []
    t = 0.0
    for step in steps_in:
        if step.get("timestamp"):
            t = (_time(step["timestamp"]) - t0).total_seconds()
        observed = {
            r.get("source_call_id"): r.get("content") or ""
            for r in (step.get("observation") or {}).get("results", [])
        }
        for call in step.get("tool_calls") or ():
            index = len(calls)
            arguments = call.get("arguments") or {}
            row = recorded[index] if index < len(recorded) else None
            digest = _sha256(
                json.dumps(call["arguments"], sort_keys=True, separators=(",", ":")).encode()
            )
            if row is None or row.arguments_sha256 != digest or row.tool != call["function_name"]:
                raise SnapshotError(f"trajectory call {index} does not hash to its recorded row")
            output = observed.get(call.get("tool_call_id"), "")
            wall = re.search(r"Wall time: ([\d.]+)", output)
            session = re.search(r"Process running with session ID (\d+)", output)
            calls.append(
                {
                    "t": t,
                    "fn": call["function_name"],
                    "cmd": str(arguments.get("cmd") or "") if isinstance(arguments, dict) else "",
                    "sid_arg": arguments.get("session_id") if isinstance(arguments, dict) else None,
                    "wall": float(wall.group(1)) if wall else 0.0,
                    "running_sid": int(session.group(1)) if session else None,
                    "exited": "Process exited" in output,
                }
            )
    # write_stdin polls belong to the call that opened their session
    origin: dict[int, int] = {}
    for i, call in enumerate(calls):
        if call["fn"] == "exec_command" and call["running_sid"] is not None:
            origin[call["running_sid"]] = i
        call["owner"] = i
        if (
            call["fn"] == "write_stdin"
            and call["sid_arg"] is not None
            and call["sid_arg"] in origin
        ):
            call["owner"] = origin[call["sid_arg"]]
    owned: dict[int, list[int]] = {}
    for i, call in enumerate(calls):
        owned.setdefault(call["owner"], []).append(i)

    def refine_step(label: str, fit: FitRecord | None, ids: Sequence[str], t_start: float, t_end: float, index: int | None, n: int) -> dict[str, Any]:  # fmt: skip
        return {
            "kind": "refine",
            "run_label": label,
            "phases": [{"candidate_id": x, "formula": library.formula.get(x)} for x in ids],
            "t_start": round(t_start, 1),
            "t_end": round(t_end, 1),
            "rwp": fit.rwp if fit else None,
            "gof": _r(fit.gof, 3) if fit else None,
            "finished_rietveld": bool(fit and fit.final),
            "calls": n,
            "call_index": index,
        }

    events: list[tuple[float, str, dict[str, Any]]] = []
    for i, call in enumerate(calls):
        if call["owner"] != i:
            continue
        members = [calls[j] for j in owned[i]]
        t_end = max(m["t"] + m["wall"] for m in members)
        label_match = (
            re.search(r"--run-label\s+(\S+)", call["cmd"]) if "refine.py" in call["cmd"] else None
        )
        if label_match:
            label = label_match.group(1).strip("'\"")
            fit = fits.get(label)
            if not any(m["exited"] for m in members) and fit and fit.elapsed_seconds is not None:
                t_end = call["t"] + fit.elapsed_seconds
            listed = re.search(r"--candidates\s+(.+?)\s+--", call["cmd"])
            ids = listed.group(1).split() if listed else (list(fit.candidates) if fit else [])
            events.append(
                (
                    call["t"],
                    "refine",
                    refine_step(label, fit, ids, call["t"], t_end, i, len(members)),
                )
            )
        elif _is_answer_write(call["cmd"]):
            events.append((call["t"], "answer", {"kind": "answer", "t": round(call["t"], 1), "calls": len(members), "call_index": i}))  # fmt: skip
        else:
            events.append((call["t"], "call", {"t": call["t"], "t_end": t_end, "phrase": _phrase(call["cmd"]), "n": len(members), "call_index": i}))  # fmt: skip
    # A saved fit with no refine.py call of its own ran from a script the agent wrote: it is
    # placed at the first call that names its run label, and has no time when none does.
    called = {p["run_label"] for _, kind, p in events if kind == "refine"}
    for label, fit in sorted(fits.items()):
        if label in called:
            continue
        naming = next(
            (i for i, c in enumerate(calls) if c["owner"] == i and label in c["cmd"]), None
        )
        t_start = calls[naming]["t"] if naming is not None else math.inf
        t_end = t_start + (fit.elapsed_seconds or 0)
        step = refine_step(label, fit, fit.candidates, t_start, t_end, naming, 0) | {
            "from_script": True
        }
        if naming is None:
            step["t_start"] = step["t_end"] = None
        events.append((t_start, "refine", step))
    for t in _compactions(rollout, t0):
        events.append((t, "compaction", {"kind": "compaction", "t": round(t, 1)}))
    events.sort(key=lambda e: e[0])

    steps: list[dict[str, Any]] = []
    for _, kind, payload in events:
        if kind == "call":
            if steps and steps[-1].get("_seg"):
                seg = steps[-1]
                seg["t_end"] = max(seg["t_end"], payload["t_end"])
                seg["calls"] += payload["n"]
                seg["_ph"].append(payload["phrase"])
            else:
                steps.append({"_seg": True, "kind": "segment", "t_start": payload["t"], "t_end": payload["t_end"], "calls": payload["n"], "_ph": [payload["phrase"]], "call_index": payload["call_index"]})  # fmt: skip
        else:
            steps.append(payload)
    # cap at MAX_STEPS: merge the longest run of segments separated only by compactions
    while len(steps) > MAX_STEPS:
        best = None
        i = 0
        while i < len(steps):
            if steps[i].get("_seg"):
                j = i
                while (
                    j + 2 < len(steps)
                    and steps[j + 1]["kind"] == "compaction"
                    and steps[j + 2].get("_seg")
                ):
                    j += 2
                if j > i and (best is None or j - i > best[1] - best[0]):
                    best = (i, j)
                i = j + 1
            else:
                i += 1
        if best is None:
            break
        i, j = best
        seg = steps[i]
        for s in steps[i + 1 : j + 1]:
            if s.get("_seg"):
                seg["t_end"] = max(seg["t_end"], s["t_end"])
                seg["calls"] += s["calls"]
                seg["_ph"] += s["_ph"]
            else:
                seg.setdefault("compactions_inside", []).append(s["t"])
        steps[i : j + 1] = [seg]
    first = True
    for s in steps:
        if s.pop("_seg", False):
            s["title"] = _title(s.pop("_ph"), first)
            first = False
            s["t_start"], s["t_end"] = round(s["t_start"], 1), round(s["t_end"], 1)
        if s["kind"] in STEP_KIND:
            s["attempt"], s["step_kind"] = attempt, STEP_KIND[s["kind"]]
        else:
            s["attempt"] = s["call_index"] = s["step_kind"] = None
    return [{k: s[k] for k in STEP_ORDER if k in s} for s in steps]


def _compactions(rollout: bytes | None, t0: datetime) -> list[float]:
    if rollout is None:
        return []
    found = []
    for line in rollout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "compacted" and row.get("timestamp"):
            found.append((_time(row["timestamp"]) - t0).total_seconds())
    return found


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- trajectory -> screenings ------------------------------------------------------------------

SCREEN_LINE = re.compile(
    r"^\s*(\d+)\.\s+(\S+)\s.*?:\s*fom\s+(-?\d+(?:\.\d+)?)\s+cov\s+(-?\d+(?:\.\d+)?)\s+s\s+[+-]?\d+(?:\.\d+)?%"
    r"\s+expl\s+(\d+(?:\.\d+)?)%\s+missed\s+(\d+)/(\d+)\s*$"
)
SCREEN_OF = re.compile(r"^Ranking \(top \d+ of (\d+)")
SCREEN_COVER = "Greedy covering set"


def screenings(trajectory: Mapping[str, Any], library: Library) -> list[dict[str, Any]]:
    """Each candidate-screen ranking (physics/screen.py) the agent read, at the call whose output
    showed it: the session poll of a long screen, or a later cat of the log it wrote. Parsed field
    by field, so the page can say how the agent found each phase; a re-read of the same ranking is
    dropped. The residual mode prints no ranking line and is never taken. Only candidate ids the
    case library holds and numbers pass; no text of the tool's output is copied, so nothing of the
    run's paths, names or prose reaches a payload (U10)."""
    outputs: list[str] = []
    for step in trajectory["steps"]:
        observed = {
            r.get("source_call_id"): r.get("content") or ""
            for r in (step.get("observation") or {}).get("results", [])
        }
        outputs += [observed.get(call.get("tool_call_id"), "") for call in step.get("tool_calls") or ()]
    found: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        ranking: list[dict[str, Any]] = []
        covering: list[str] = []
        ranked_of: int | None = None
        for line in output.splitlines():
            if (m := SCREEN_LINE.match(line)) and m.group(2) in library.formula:
                ranking.append(
                    {
                        "rank": int(m.group(1)),
                        "candidate_id": m.group(2),
                        "fom": float(m.group(3)),
                        "cov": float(m.group(4)),
                        "expl_percent": float(m.group(5)),
                        "missed_lines": [int(m.group(6)), int(m.group(7))],
                    }
                )
            elif m := SCREEN_OF.match(line):
                ranked_of = int(m.group(1))
            elif line.startswith(SCREEN_COVER) and ":" in line:
                picks = (part.split() for part in line.split(":", 1)[1].split(";"))
                covering = [t[0] for t in picks if t and t[0] in library.formula]
        seen = [(x["ranking"], x["covering_set"]) for x in found]
        if (ranking or covering) and (ranking, covering) not in seen:
            found.append(
                {"call_index": index, "ranked_of": ranked_of, "ranking": ranking, "covering_set": covering}
            )
    return found


# --- evidence of one attempt --------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """Everything the snapshot and the blind payload show about one attempt, derived once."""

    kind: SnapshotKind
    case: CaseManifest
    root: str
    origin: str
    answered: bool
    fit: FitRecord | None
    fit_role: str
    summary: FitSummary | None
    curve_sha256: str | None
    curve: np.ndarray | None
    dia: Dia | None
    answer_percent: Mapping[str, float | None] | None
    answer_phases: tuple[str, ...]
    claims: tuple[tuple[str, str], ...]  # (kind, text)
    reported: Mapping[str, float | None]  # the agent's own rwp and gof, when it reported them
    descriptors: Mapping[str, Mapping[str, str]]  # candidate id -> SnapshotPhase reported_* field
    quantity_basis: JsonValue
    submitted: JsonValue  # AnalysisSnapshot.reported_answer
    checks: tuple[dict[str, Any], ...]
    lattice: tuple[dict[str, Any], ...]
    steps: tuple[dict[str, Any], ...]
    missing: tuple[str, ...]
    library: Library
    screenings: tuple[dict[str, Any], ...] = ()  # screenings(): ids and numbers only


def _claims(analysis: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """The agent's own statements, verbatim, in document order."""
    if not analysis:
        return ()
    found: list[tuple[str, str]] = []

    def walk(value: object, kind: str) -> None:
        if isinstance(value, str):
            if len(value.split()) >= MIN_CLAIM_WORDS:
                found.append((kind, value))
        elif isinstance(value, dict):
            for key, item in value.items():
                if key != "definition" and not key.endswith(("unit", "units")):
                    walk(item, kind)
        elif isinstance(value, list):
            for item in value:
                walk(item, kind)

    for key, value in analysis.items():
        if key in NOT_CLAIMS:
            continue
        kind = CLAIM_KINDS.get(key, key)
        if key == "phases":
            for phase in value or ():
                if isinstance(phase, dict):
                    walk(phase.get("evidence"), kind)
        else:
            walk(value, kind)
    return tuple(found)


def _descriptors(analysis: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """Each answered phase's own formula, space group, crystal system and name, verbatim."""
    found: dict[str, dict[str, str]] = {}
    for phase in (analysis or {}).get("phases") or ():
        if isinstance(phase, dict) and isinstance(phase.get("candidate_id"), str):
            found[phase["candidate_id"]] = {
                field: str(phase[key])
                for key, field in REPORTED_FIELDS.items()
                if phase.get(key) is not None
            }
    return found


def _find_stat(
    analysis: Mapping[str, Any] | None, paths: Sequence[tuple[str, ...]]
) -> float | None:
    """First number in the answer's free-form fit_statistics under any of the given key paths."""
    stats = (analysis or {}).get("fit_statistics") or {}
    for path in paths:
        current: object = stats
        for key in path:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            return float(current)
    return None


def _answer_file(store: Any, outcome: Completed, end: ImportedSubmitted) -> dict[str, Any] | None:
    """The answer file the import bound by digest, when it is one of the import's receipts and
    names the phases the import parsed from it. Otherwise None: no other receipt stands in, so
    a draft that reports the same phases and amounts is never shown as the answer."""
    sha = end.answer_sha256
    if sha is None or sha not in {ref.sha256 for ref in outcome.origin.receipts}:
        return None
    try:
        value = json.loads(store.read_blob(sha))
    except ValueError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("phases"), list):
        return None
    named = tuple(p.get("candidate_id") if isinstance(p, dict) else None for p in value["phases"])
    return value if named == end.answer.phases else None


def _run_names(attempt: ChildAttempt, result: ImportedRunResult) -> tuple[str, ...]:
    """The strings that name the run: its job, trial, model id and its words of four or more
    letters (e.g. qwen, astra), and harness. Longest first, so a model id goes before its words."""
    model = result.agent.model_id
    harbor = attempt.action.harbor
    words = re.findall(r"[A-Za-z]{4,}", model)
    found = {
        harbor.job_name,
        harbor.trial_id,
        model,
        model.rsplit("/", 1)[-1],
        result.agent.harness,
        *words,
    }
    return tuple(sorted((t for t in found if t), key=lambda t: (-len(t), t)))


def _scrubbed(value: Any, identity: Sequence[str]) -> Any:
    """`value` with every absolute path and every whole-word mention of the run replaced by a
    marker. Keys, numbers, nulls, booleans and order are kept as they are."""
    if isinstance(value, str):
        text = ABSOLUTE_PATH.sub(PATH_MARK, value)
        for token in identity:
            text = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
                IDENTITY_MARK,
                text,
                flags=re.I,
            )
        return text
    if isinstance(value, dict):
        return {_scrubbed(k, identity): _scrubbed(v, identity) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrubbed(v, identity) for v in value]
    return value


def _guard(attempt: ChildAttempt) -> Completed:
    outcome = attempt.outcome
    if not isinstance(outcome, Completed):
        raise SnapshotError(f"the attempt's outcome is {outcome.status}, not completed")
    if outcome.contamination is not None or outcome.custody_scan == "match":
        raise SnapshotError("the attempt carries custody bytes")
    refs = (*outcome.artifacts, *getattr(outcome.origin, "receipts", ()))
    if any(ref.role == "custody_withheld" for ref in refs):
        raise SnapshotError("the attempt carries custody-withheld bytes")
    return outcome


def _imported_evidence(store: Any, path: Sequence[tuple[str, Any]], library: Library) -> Evidence:
    root_digest, root = path[0]
    origin, attempt = path[-1]
    if not isinstance(attempt.action, ImportRun):
        raise SnapshotError("an agent_answer snapshot freezes an imported run")
    outcome = _guard(attempt)
    result = outcome.result
    assert isinstance(result, ImportedRunResult)
    receipts = read_receipts(store, outcome.origin.receipts)
    # A refinement finished in other software (importers.external): its summary is the fit, and
    # its answer is that fit's own phase list, so no check may compare the two. Keyed on the
    # field only the external importer sets, never on the run-supplied job name.
    external = attempt.action.external_source is not None
    fits = {
        label: _recorded_fit(label, value, receipts.requests.get(label))
        for label, value in sorted(receipts.results.items())
    }
    missing = (
        list(result.completeness.missing) if isinstance(result.completeness, Incomplete) else []
    )
    answered = isinstance(result.end, ImportedSubmitted)
    answer = result.end.answer if isinstance(result.end, ImportedSubmitted) else None
    summary = result.basis_fit
    fit = curve = dia = None
    curve_sha = None
    cited = summary is not None and summary.curve is not None
    if cited:
        role = "cited by the answer"
        curve_sha = summary.curve.sha256
        curve = read_curve(store.read_blob(curve_sha))
        dia = _dia_for_curve(receipts.dias, curve) or next(
            (d for data in receipts.unlabeled if (d := _dia_matching(data, curve)) is not None), None
        )
        if dia is not None and dia.label in fits and _matches(fits[dia.label], summary):
            fit = fits[dia.label]
        else:
            fit = next((f for f in fits.values() if _matches(f, summary)), None)
        if fit is None and external:
            fit = _summary_fit(summary)
            if dia is None and curve.shape[1] >= 5:
                # No .dia, but the curve keeps the software's own background (GSAS-II): the
                # residual check reads it; there are no per-phase columns.
                dia = Dia(None, None, (), curve[:, (0, 1, 2, 4)])
        if dia is not None:
            dia = _named_by_candidates(dia, [p.candidate_id for p in summary.phases])
        if fit is not None:  # the esd is on the .lst Q line, which the importer parsed
            esd = {p.candidate_id: p.weight_percent_error for p in summary.phases}
            fit = replace(fit, phases=tuple(p | {"esd": esd.get(p["id"])} for p in fit.phases))
    else:
        role = "lowest-Rwp finished Rietveld fit (no cited fit)"
        done = [f for f in fits.values() if f.final and not f.failed and f.rwp is not None]
        if done:
            fit = min(done, key=lambda f: (f.rwp, f.label))
            data = receipts.dias.get((fit.label, "rietveld"))
            dia = read_dia(data) if data else None
            curve_bytes = _curve_for_dia(receipts.curves, dia) if dia else None
            if curve_bytes is not None:
                curve_sha = _sha256(curve_bytes)
                curve = read_curve(curve_bytes)
    if fit is None:
        missing.append("fit")
    elif dia is None:
        missing.append("fit.dia")
    analysis = None
    if isinstance(result.end, ImportedSubmitted):
        found = _answer_file(store, outcome, result.end)
        if found is None:
            missing.append("answer text")
        else:
            analysis = _scrubbed(found, _run_names(attempt, result))
    lattice = tuple(lattice_rows(fit, library)) if fit else ()
    answer_percent = dict(answer.reported_weight_percent) if answer else None
    checks = _checks(fit, dia, lattice, answer_percent if cited else None)
    if external:
        source = attempt.action.external_source
        replaced = {
            "K5": _check(
                "K5",
                "Answer equals the cited fit",
                "not_computable",
                "Not applicable: the answer is the imported refinement's own phase list and amounts, "
                f"so there is no independent fit to compare it with (source {source}).",
            )
        }
        if fit is not None and all(p.get("wt") is None for p in fit.phases):
            # K3 reads phases of at least 2 wt%; with no amounts it would pass over nothing.
            stage = summary.stage if summary is not None else "imported"
            replaced["K3"] = _check(
                "K3",
                "Lattice shift from the CIF cell",
                "not_computable",
                f"Not applicable: the imported {stage} fit carries no phase amounts (a Pawley or Le Bail "
                "fit extracts intensities without them), and the lattice-shift check reads only phases "
                f"of at least 2 wt%, so it has nothing to check (source {source}).",
            ) | {"advisory": True, "phases": list(lattice)}
        checks = tuple(replaced.get(check["code"], check) for check in checks)
    trajectory = receipts.trajectory
    if trajectory is None:
        missing.append("steps")
    steps = (
        build_steps(trajectory, result.trajectory, fits, library, origin, receipts.rollout)
        if trajectory
        else []
    )
    reported = {
        "rwp": _find_stat(
            analysis, [("rwp", "value"), ("rietveld", "rwp_percent"), ("rwp_percent",), ("rwp",)]
        ),
        "gof": _find_stat(
            analysis, [("gof", "value"), ("rietveld", "gof_dimensionless"), ("gof",)]
        ),
    }
    return Evidence(
        screenings=tuple(screenings(trajectory, library)) if trajectory else (),
        kind="agent_answer",
        case=root.case,
        root=root_digest,
        origin=origin,
        answered=answered,
        fit=fit,
        fit_role=role,
        summary=summary,
        curve_sha256=curve_sha,
        curve=curve,
        dia=dia,
        answer_percent=answer_percent,
        answer_phases=answer.phases if answer else (),
        claims=_claims(analysis),
        reported=reported,
        descriptors=_descriptors(analysis),
        quantity_basis=(analysis or {}).get("quantity_basis"),
        submitted=None
        if analysis is None
        else {k: v for k, v in analysis.items() if k not in NOT_SUBMITTED},
        checks=checks,
        lattice=lattice,
        steps=tuple(steps),
        missing=tuple(missing),
        library=library,
    )


def _correction_evidence(store: Any, path: Sequence[tuple[str, Any]], library: Library) -> Evidence:
    """An expert's refit child submitted as an answer (KTD10): its own fit and its own steps,
    from the first step after the imported run. The imported run's identity is never named."""
    root_digest, root = path[0]
    origin, attempt = path[-1]
    if not isinstance(attempt.action, Refine) or attempt.actor.kind != "human":
        raise SnapshotError("an expert_correction snapshot freezes a human's refine")
    outcome = _guard(attempt)
    result = outcome.result
    if not isinstance(result, RefineResult) or result.final is None:
        raise SnapshotError("the refit has no final fit")
    summary = result.final
    curve_ref = next(
        (a for a in outcome.artifacts if a.role == "fit_curve" and a.stage == summary.stage),
        summary.curve,
    )
    dia_ref = next(
        (a for a in outcome.artifacts if a.role == "native_dia" and a.stage == summary.stage), None
    )
    curve = read_curve(store.read_blob(curve_ref.sha256)) if curve_ref else None
    dia = read_dia(store.read_blob(dia_ref.sha256)) if dia_ref else None
    fit = _live_fit(f"refit {origin[:12]}", result)
    lattice = tuple(lattice_rows(fit, library))
    checks = list(_checks(fit, dia, lattice, None))
    checks[3]["detail"] = (
        "No agent answer cites this fit: the expert's refit is the proposed answer itself."
    )
    steps = []
    for digest, step in path[1:]:
        if isinstance(step.action, ImportRun):
            continue
        steps.append(_attempt_step(digest, step, library))
    missing = [] if dia is not None else ["fit.dia"]
    return Evidence(
        kind="expert_correction",
        case=root.case,
        root=root_digest,
        origin=origin,
        answered=True,
        fit=fit,
        fit_role="expert refit submitted as an answer",
        summary=summary,
        curve_sha256=curve_ref.sha256 if curve_ref else None,
        curve=curve,
        dia=dia,
        answer_percent=None,
        answer_phases=tuple(p.candidate_id for p in summary.phases),
        claims=(),
        reported={"rwp": None, "gof": None},
        descriptors={},
        quantity_basis=None,
        submitted=None,
        checks=tuple(checks),
        lattice=lattice,
        steps=tuple(steps),
        missing=tuple(missing),
        library=library,
    )


def _attempt_step(digest: str, attempt: ChildAttempt, library: Library) -> dict[str, Any]:
    action = attempt.action
    if isinstance(action, AddPhases):
        title = "Added " + ", ".join(f"{c} ({library.formula.get(c)})" for c in action.candidates)
    elif isinstance(action, RemovePhases):
        title = "Removed " + ", ".join(f"{c} ({library.formula.get(c)})" for c in action.candidates)
    elif isinstance(action, ReplacePhase):
        title = f"Replaced a phase with {action.model_dump(mode='json')}"
    else:
        title = action.verb.replace("_", " ").capitalize()
    step: dict[str, Any] = {"kind": "attempt", "title": title}
    outcome = attempt.outcome
    if (
        isinstance(action, Refine)
        and isinstance(outcome, Completed)
        and isinstance(outcome.result, RefineResult)
    ):
        final = outcome.result.final
        step |= {
            "rwp": final.rwp_percent if final else None,
            "gof": _r(final.gof, 3) if final and final.gof is not None else None,
            "phases": [
                {"candidate_id": p.candidate_id, "formula": library.formula.get(p.candidate_id)}
                for p in (final.phases if final else ())
            ],  # fmt: skip
            "seconds": round(outcome.elapsed_seconds, 1),
        }
    return step | {"attempt": digest, "call_index": None, "step_kind": "attempt"}


def _checks(
    fit: FitRecord | None,
    dia: Dia | None,
    lattice: Sequence[dict[str, Any]],
    answer: Mapping[str, float | None] | None,
) -> tuple[dict[str, Any], ...]:
    if fit is None:
        none = "No finished fit to check."
        return (
            _check("K1", "Rietveld degraded after a good Pawley fit", "not_computable", none),
            _check("K2", "Unexplained residual peaks", "not_computable", none)
            | {"max_pct": None, "peaks": []},
            _check("K3", "Lattice shift from the CIF cell", "not_computable", none)
            | {"advisory": True, "phases": []},
            _check("K5", "Answer equals the cited fit", "not_computable", none),
        )
    residual = (
        k2(dia)
        if dia is not None
        else _check(
            "K2",
            "Unexplained residual peaks",
            "not_computable",
            "The fit's final-stage .dia is not in the store.",
        )
        | {"max_pct": None, "peaks": []}
    )
    return (k1(fit), residual, k3(lattice), k5(answer, fit))


def _derive(store: Any, path: Sequence[tuple[str, Any]], kind: SnapshotKind) -> Evidence:
    root = path[0][1]
    assert isinstance(root, RootAttempt)
    library = _library(store, root.case)
    if kind == "agent_answer":
        return _imported_evidence(store, path, library)
    if kind == "expert_correction":
        return _correction_evidence(store, path, library)
    raise SnapshotError(f"no snapshot builder for kind {kind}")


_EVIDENCE: dict[tuple[Any, str, SnapshotKind], Evidence] = {}
EVIDENCE_CACHE_SIZE = 64


def evidence(store: Any, principal: Principal, origin: str, kind: SnapshotKind) -> Evidence:
    """Authorized on every call, derived once per (store, attempt, kind): attempts and blobs
    are immutable."""
    path = store.lineage(principal, origin)
    key = (store, origin, kind)
    if key not in _EVIDENCE:
        if len(_EVIDENCE) >= EVIDENCE_CACHE_SIZE:
            _EVIDENCE.pop(next(iter(_EVIDENCE)))
        _EVIDENCE[key] = _derive(store, path, kind)
    return _EVIDENCE[key]


# --- snapshot -----------------------------------------------------------------------------------


def build_snapshot(
    store: Any, principal: Principal, origin: AttemptRef, kind: SnapshotKind
) -> AnalysisSnapshot:
    """Derive the snapshot of one attempt from the store: its state's fit and curve blob, its
    imported answer, the K checks over the curve. Pure over store reads; writes nothing.
    Raises when the attempt carries custody bytes or a reference field."""
    ev = evidence(store, principal, origin, kind)
    fit_esd = {p["id"]: p.get("esd") for p in ev.fit.phases} if ev.fit else {}
    fit_wt = {p["id"]: p.get("wt") for p in ev.fit.phases} if ev.fit else {}
    shift = {row["candidate_id"]: row["max_abs_axis_shift_pct"] for row in ev.lattice}
    if ev.answered and ev.answer_phases:
        weights = ev.answer_percent if ev.answer_percent is not None else fit_wt
        ids = ev.answer_phases  # the answer's own weights; an expert correction's are its fit's
    else:
        weights, ids = fit_wt, tuple(fit_wt)
    phases = tuple(
        SnapshotPhase(
            phase_id=cid,
            formula=ev.library.formula.get(cid),
            space_group=ev.library.space_group(cid),
            weight_percent=_r(weights.get(cid), 4),
            weight_percent_error=_r(fit_esd.get(cid), 4),
            lattice_shift_percent=shift.get(cid),
            **ev.descriptors.get(cid, {}),
        )
        for cid in ids
    )
    k2_check = ev.checks[1]
    peaks = tuple(
        ResidualPeak(
            peak_id=p["peak_id"],
            two_theta_deg=p["two_theta"],
            height_percent=p["pct_of_strongest_peak"],
            residual_counts=p["residual_counts"],
        )
        for p in k2_check["peaks"]
    )
    summary = ev.summary
    stage = summary.stage if summary else ("rietveld" if ev.fit and ev.fit.final else None)
    return AnalysisSnapshot(
        kind=kind,
        origin=origin,
        root=ev.root,
        case_id=ev.case.case_id,
        pattern_sha256=ev.case.pattern.sha256,
        fit_curve_sha256=ev.curve_sha256,
        stage=stage,
        rwp_percent=summary.rwp_percent if summary else (ev.fit.rwp if ev.fit else None),
        gof=summary.gof if summary else (ev.fit.gof if ev.fit else None),
        quantification=summary.quantification if summary else None,
        phases=phases,
        residual_peaks=peaks,
        checks=tuple(
            SnapshotCheck(
                **{k: c[k] for k in ("code", "name", "status", "detail", "policy_version")}
            )
            for c in ev.checks
        ),
        claims=tuple(text for _, text in ev.claims),
        reported_quantity_basis=ev.quantity_basis,
        reported_answer=ev.submitted,
        answered=ev.answered,
        chemistry=_chemistry(store, phases, ev.case),
    )


def _chemistry(
    store: Any, phases: Sequence[SnapshotPhase], case: CaseManifest
) -> ChemistryEvidence:
    library = {c.candidate_id: c for c in case.candidates}
    inputs = [
        chemistry.candidate_phase(
            library[p.phase_id],
            store.read_blob(library[p.phase_id].cif.sha256).decode(errors="replace"),
        )
        for p in phases
    ]
    return chemistry.chemistry_evidence(inputs, chemistry.case_precursors(case))


# --- the page's payloads ------------------------------------------------------------------------


def _downsample(curve: np.ndarray) -> tuple[int, dict[str, list[float | None]]]:
    stride = max(1, math.ceil(len(curve) / MAX_POINTS))
    rows = curve[::stride]
    names = ("two_theta", "observed", "calculated", "difference")
    return stride, {name: [_sig4(x) for x in rows[:, i]] for i, name in enumerate(names)}


def _brief(store: Any, case: CaseManifest, missing: list[str]) -> dict[str, Any]:
    """The case brief from the package manifest the agent read (its actor-visible input)."""
    source = next(s for s in case.ingest.sources if s.original_name == "manifest.json")
    entry = json.loads(store.read_blob(source.blob.sha256))["cases"][0]
    context = entry["material_context"]
    measurement = dict(entry["measurement"])
    profile = entry["instrument_profile"]
    precursors = [
        {"formula": p["formula"], "mass_g": p.get("actual_mass_g")}
        for p in context.get("precursors", [])
    ]
    if any(p["mass_g"] is None for p in precursors):
        missing.append("brief.precursors[].mass_g")
    processing = []
    for step in context.get("processing", []):
        row = {"action": step.get("action")}
        for key in ("temperature_c", "duration_min", "atmosphere", "furnace"):
            row[key] = step.get(key)
            if step.get("action") == "Heating" and step.get(key) is None:
                missing.append(f"brief.processing[Heating].{key}")
        other = {k: v for k, v in step.items() if k not in row and not isinstance(v, (dict, list))}
        if other:
            row["other"] = other
        processing.append(row)
    target = context.get("declared_target")
    if target is None:
        missing.append("brief.declared_target")
    caveat = measurement.pop("profile_caveat", None)
    if caveat is None:
        missing.append("brief.profile_caveat")
    acquisition = dict(measurement)
    try:
        span = measurement["two_theta_end"] - measurement["two_theta_start"]
        acquisition["step_deg_derived"] = round(span / (measurement["point_count"] - 1), 6)
        acquisition["step_note"] = "derived: (two_theta_end - two_theta_start) / (point_count - 1); the manifest has no step field"  # fmt: skip
    except (KeyError, ZeroDivisionError, TypeError):
        missing.append("brief.acquisition.step")
    return {
        "case_id": entry["case_id"],
        "elements": entry.get("chemistry_elements"),
        "specimen": context.get("specimen"),
        "precursors": precursors,
        "precursor_amounts_note": WITHHELD_AMOUNTS
        if str(context.get("precursor_amounts")).startswith("withheld")
        else context.get("precursor_amounts"),
        "processing": processing,
        "declared_target": None
        if target is None
        else {
            "formula": target.get("formula"),
            "stoichiometry": target.get("stoichiometry"),
        },  # fmt: skip
        "instrument_profile": profile.get("profile"),
        "instrument_profile_confidence": profile.get("instrument_profile_confidence"),
        "quantitative_claims": profile.get("quantitative_claims"),
        "profile_caveat": caveat,
        "acquisition": acquisition,
        "package_manifest_sha256": source.blob.sha256,
    }


def _residual_windows(ev: Evidence, snap: AnalysisSnapshot) -> list[dict[str, Any]]:
    """The fit at native resolution and full precision within RESIDUAL_WINDOW_DEG of each listed
    residual peak, with background and per-phase contributions when the .dia is stored: the
    peak-preserving evidence the strided curve loses, shown to the rater and the judge alike."""
    rows = ev.dia.rows if ev.dia is not None else ev.curve
    if rows is None:
        return []
    return [
        {"peak_id": peak.peak_id} | _window(rows, peak.two_theta_deg, ev.dia)
        for peak in snap.residual_peaks
    ]


def _phase_windows(
    dia: Dia | None, phase_ids: Sequence[str], missing: list[str]
) -> list[dict[str, Any]]:
    """Native windows around each reported phase's own calculated peaks (PHASE_WINDOW_RULE), so a
    phase that leaves no residual still shows observed against model where it puts intensity.
    Chosen from the fit's .dia alone, never from a reference or a known angle. A reported phase
    the .dia has no contribution for gets no window and a missing entry, never a drawn curve."""
    if dia is None:
        return []  # the view already lists "fit" or "fit.dia" as missing
    tt = dia.rows[:, 0]
    windows = []
    for cid in phase_ids:
        if cid not in dia.phase_ids:
            missing.append(f"fit.phase_curves.{cid}")
            continue
        contribution = dia.rows[:, 4 + dia.phase_ids.index(cid)]
        maxima = [i for i in find_peaks(contribution)[0] if contribution[i] > 0]
        centers: list[float] = []
        for i in sorted(maxima, key=lambda i: (-contribution[i], tt[i])):
            if len(centers) == PHASE_WINDOWS:
                break
            if all(abs(tt[i] - c) > 2 * RESIDUAL_WINDOW_DEG for c in centers):
                centers.append(float(tt[i]))
        windows += [{"phase_id": cid, "center_deg": c} | _window(dia.rows, c, dia) for c in centers]
    return windows


def _window(rows: np.ndarray, center: float, dia: Dia | None) -> dict[str, Any]:
    """Every native row within RESIDUAL_WINDOW_DEG of center at full precision, with background
    and each phase's contribution when the .dia is stored."""
    near = rows[np.abs(rows[:, 0] - center) <= RESIDUAL_WINDOW_DEG]
    window: dict[str, Any] = {
        "two_theta_deg": near[:, 0].tolist(),
        "observed": near[:, 1].tolist(),
        "calculated": near[:, 2].tolist(),
        "difference": (near[:, 1] - near[:, 2]).tolist(),
    }
    if dia is not None:
        window["background"] = near[:, 3].tolist()
        window["phase_curves"] = {
            cid: near[:, 4 + j].tolist() for j, cid in enumerate(dia.phase_ids)
        }
    return window


def _phase_curves(dia: Dia, stride: int) -> dict[str, list[float | None]]:
    """Each phase's calculated net contribution: the .dia columns after background."""
    rows = dia.rows[::stride]
    return {cid: [_sig4(x) for x in rows[:, 4 + j]] for j, cid in enumerate(dia.phase_ids)}


def blind_payload(
    store: Any, labels: Any, principal: Principal, snapshot: Digest
) -> dict[str, Any]:
    """The page's data for one snapshot before sign-off: the D v3 episode shape (brief, answer,
    fit with two_theta/observed/calculated/difference/phase_curves, checks, steps, missing)
    plus digest, root, origin, kind, residual_peaks and chemistry. Never carries job, model,
    harness, series, reference_match or sibling episodes."""
    snap: AnalysisSnapshot = labels.snapshot(principal, snapshot)
    return _blind(store, evidence(store, principal, snap.origin, snap.kind), snap, snapshot)


def attempt_payload(store: Any, principal: Principal, attempt: AttemptRef) -> dict[str, Any]:
    """The blind payload of an attempt that is not a frozen snapshot, such as a refit child shown
    beside its parent (D10). Same shape and the same blindness: nothing names the run, also when
    the attempt descends from an imported one. A refit that no expert submitted is no answer.
    store.lineage applies the rater projection, so before the reveal a rater reviews only its own
    refit, never another rater's or one built on another's (CR-07)."""
    path = store.lineage(principal, attempt)
    action = path[-1][1].action if len(path) > 1 else None
    if isinstance(action, ImportRun):
        kind: SnapshotKind = "agent_answer"
    elif isinstance(action, Refine):
        kind = "expert_correction"
    else:
        raise SnapshotError("only an imported run or a refit has a fit to review")
    snap = build_snapshot(store, principal, attempt, kind)
    payload = _blind(store, evidence(store, principal, attempt, kind), snap, None)
    if kind == "expert_correction":
        payload |= {"kind": "refit", "answered": False, "answer": None}
    return payload


def _blind(
    store: Any, ev: Evidence, snap: AnalysisSnapshot, snapshot: Digest | None
) -> dict[str, Any]:
    missing = list(ev.missing)
    brief = _brief(store, ev.case, missing)
    # The answer's own words for each phase; the candidate CIF's metadata sits apart under "cif".
    # An expert correction answers with the candidates themselves, so its words are the CIF's.
    correction = snap.kind == "expert_correction"
    phases = [
        {
            "candidate_id": p.phase_id,
            "formula": p.formula if correction else p.reported_formula,
            "space_group": p.space_group if correction else p.reported_space_group,
            "crystal_system": p.reported_crystal_system,
            "name": p.reported_name,
            "cif": {"formula": p.formula, "space_group": p.space_group},
            "weight_percent": _r(p.weight_percent),
            "esd": _r(p.weight_percent_error, 3),
            "lattice_shift_pct": p.lattice_shift_percent,
        }
        for p in snap.phases
    ]
    answer = None
    if snap.answered:
        answer = {
            "cited_fit": ev.fit.label if ev.fit else None,
            "phases": phases,
            "rwp": ev.reported["rwp"] if snap.kind == "agent_answer" else snap.rwp_percent,
            "gof": ev.reported["gof"] if snap.kind == "agent_answer" else _r(snap.gof, 4),
            "verdict": "not recorded" if snap.kind == "agent_answer" else "expert correction",
            "quantity_basis": snap.reported_quantity_basis,
            "submitted": snap.reported_answer,
            "esd_source": ESD_SOURCE,
            "lattice_shift_source": SHIFT_SOURCE,
            "claims": [{"kind": kind, "text": text} for kind, text in ev.claims],
        }
    fit = None
    if ev.fit is not None and ev.curve is not None:
        stride, curve = _downsample(ev.curve)
        fit = {
            "run_label": ev.fit.label,
            "role": ev.fit_role,
            "stage": snap.stage,
            "rwp": snap.rwp_percent,
            "gof": _r(snap.gof, 4),
            "phases": [p["id"] for p in ev.fit.phases],
            "phase_detail": [
                {
                    "candidate_id": p["id"],
                    "formula": ev.library.formula.get(p["id"]),
                    "space_group": ev.library.space_group(p["id"]),
                    "weight_percent": _r(p.get("wt")),
                    "esd": _r(p.get("esd"), 3),
                }
                for p in ev.fit.phases
            ],
            "stride": stride,
            "points": len(curve["two_theta"]),
            "native_points": len(ev.curve),
            "precision": CURVE_PRECISION,
            "curve_sha256": snap.fit_curve_sha256,
            **curve,
            "phase_curves": _phase_curves(ev.dia, stride) if ev.dia is not None else {},
        }
    phase_windows = _phase_windows(ev.dia, [p.phase_id for p in snap.phases], missing)
    return {
        "schema_version": "review-payload.v1",
        "digest": snapshot,
        "root": snap.root,
        "origin": snap.origin,
        "kind": snap.kind,
        "case_id": snap.case_id,
        "answered": snap.answered,
        "times": "seconds from the first step of the trajectory",
        "brief": brief,
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "formula": ev.library.formula.get(c.candidate_id),
                "space_group": ev.library.space_group(c.candidate_id),
            }  # fmt: skip
            for c in ev.case.candidates
        ],
        "answer": answer,
        "fit": fit,
        "checks": list(ev.checks),
        "residual_peaks": [p.model_dump(mode="json") for p in snap.residual_peaks],
        "residual_windows": _residual_windows(ev, snap),
        "phase_windows": phase_windows,
        "phase_window_rule": PHASE_WINDOW_RULE,
        "steps": list(ev.steps),
        "screenings": list(ev.screenings),
        "missing": missing,
        "chemistry": None if snap.chemistry is None else snap.chemistry.model_dump(mode="json"),
    }


# --- the analysis files, after the rater's own sign-off ------------------------------------------


def _scrub_text(data: bytes, identity: Sequence[str]) -> bytes:
    return str(_scrubbed(data.decode("utf-8", "replace"), identity)).encode()


def analysis_files(store: Any, labels: Any, principal: Principal, snapshot: Digest) -> bytes:
    """The raw files of one snapshot's analysis as a zip, for the rater's own records beside the
    page's analysis report (AXA-43zwk.18): the observed pattern, the CIF of every answered or
    fitted phase, and the cited fit's BGMN request, result and final-stage .dia. Gated as
    revealed_payload: PermissionError until this rater signed off the pattern's cohort. Nothing
    comes from custody (the receipts refuse custody-withheld bytes), the trajectory and rollout
    stay out (they name the run), and every text file passes through _scrubbed with the run's
    names, as the answer file does, so absolute paths and run names read as markers."""
    snap: AnalysisSnapshot = labels.snapshot(principal, snapshot)
    if not labels.reveal_state(principal, snap.root).signed_off:
        raise PermissionError("the analysis files open after this rater's sign-off")
    path = store.lineage(principal, snap.origin)
    ev = evidence(store, principal, snap.origin, snap.kind)
    files: dict[str, bytes] = {}
    case = ev.case
    files[f"pattern/{snap.case_id}.xy"] = store.read_blob(case.pattern.sha256)
    wanted = list(dict.fromkeys([p.phase_id for p in snap.phases] + [p["id"] for p in (ev.fit.phases if ev.fit else ())]))
    for entry in case.candidates:
        if entry.candidate_id in wanted:
            files[f"cif/{entry.candidate_id}.cif"] = store.read_blob(entry.cif.sha256)
    origin, attempt = path[-1]
    note = "The cited fit's BGMN files are not stored for this attempt."
    if isinstance(attempt.action, ImportRun) and ev.fit is not None:
        outcome = _guard(attempt)
        result = outcome.result
        assert isinstance(result, ImportedRunResult)
        identity = _run_names(attempt, result)
        receipts = read_receipts(store, outcome.origin.receipts)
        label = ev.fit.label
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        if label in receipts.requests:
            files[f"bgmn/{safe}/request.json"] = _scrub_text(json.dumps(receipts.requests[label], indent=1).encode(), identity)
        if label in receipts.results:
            files[f"bgmn/{safe}/result.json"] = _scrub_text(json.dumps(receipts.results[label], indent=1).encode(), identity)
        if (label, "rietveld") in receipts.dias:
            files[f"bgmn/{safe}/rietveld.dia"] = _scrub_text(receipts.dias[(label, "rietveld")], identity)
        note = f"bgmn/{safe}/ holds the cited fit's refinement request, its result and its final-stage .dia."
    readme = (
        f"PhaseEval analysis files: {snap.case_id}, snapshot {snapshot}.\n"
        f"pattern/ holds the observed pattern; cif/ the CIF of every answered or fitted phase.\n{note}\n"
        "Absolute paths and run names in text files read [path] and [run identity].\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in [("README.txt", readme.encode()), *sorted(files.items())]:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


# --- after sign-off ---------------------------------------------------------------------------


def revealed_payload(
    store: Any, labels: Any, principal: Principal, snapshot: Digest
) -> dict[str, Any]:
    """Run identity, this snapshot's triage route and, once every assigned rater signed off,
    the reference comparison and sibling episodes. Raises PermissionError until the store holds
    this rater's sign-off. The route is None until the rater signed off this snapshot itself
    (Triage.revealed), or when no sweep routed it under POLICY.

    W3 view: "signed off" is RevealState.signed_off, this rater's sign-off on every snapshot of
    the root's review cohort, so no identity of one analysis biases the rater's call on another
    of the same pattern. The siblings are that cohort (D14): the reveal compares what the rater
    reviewed, never an episode the rater was not shown."""
    snap: AnalysisSnapshot = labels.snapshot(principal, snapshot)
    state = labels.reveal_state(principal, snap.root)
    if not state.signed_off:
        raise PermissionError("the reveal opens after this rater's sign-off")
    path = store.lineage(principal, snap.origin)
    route = Triage(store, labels, POLICY).revealed(principal, snapshot)
    full = state.revealed
    reference = _reference(snap.case_id) if full else None
    this_import = next(
        (d for d, a in path if isinstance(a, ChildAttempt) and isinstance(a.action, ImportRun)),
        None,
    )
    siblings = []
    if full:
        for digest in labels.cohort(principal, snap.root):
            other: AnalysisSnapshot = labels.snapshot(principal, digest)
            row = _identity(store.lineage(principal, other.origin)[-1][1]) | {
                "episode_id": other.origin,
                "snapshot": digest,
                "this": digest == snapshot,
                "phases": [
                    {"candidate_id": p.phase_id, "formula": p.formula, "space_group": p.space_group}
                    for p in other.phases
                ],
                "residual_max_pct": max(
                    (p.height_percent for p in other.residual_peaks), default=0.0
                ),
            }
            if reference is not None:
                ids = tuple(p.phase_id for p in other.phases)
                basis = "answer" if other.answered else "best_fit"
                row["reference_match"] = _match(ids, reference, basis) if ids else None
            siblings.append(row)
    identity = None
    if snap.kind == "agent_answer":
        identity = _identity(path[-1][1])
    elif this_import is not None:
        corrected = next(a for d, a in path if d == this_import)
        identity = {
            "corrected_episode": _identity(corrected),
            "by": path[-1][1].actor.model_dump(mode="json"),
        }
    return {
        "schema_version": "revealed-payload.v1",
        "digest": snapshot,
        "root": snap.root,
        "kind": snap.kind,
        "reveal": state.model_dump(mode="json"),
        "identity": identity,
        "route": None if route is None else route.model_dump(mode="json"),
        "reference_revealed": full,
        "reference": reference,
        "reference_match": _match(
            tuple(p.phase_id for p in snap.phases),
            reference,
            "answer" if snap.answered else "best_fit",
        )
        if reference
        else None,
        "siblings": siblings if full else None,
        "labels": [
            label.model_dump(mode="json") for label in labels.visible_labels(principal, snapshot)
        ],
        # this rater's own sign-off on this snapshot (None for a snapshot it did not sign itself)
        "signoff": None if (own := labels.own_signoff(principal, snapshot)) is None else own.model_dump(mode="json"),
    }


def _identity(attempt: ChildAttempt) -> dict[str, Any]:
    result = attempt.outcome.result if isinstance(attempt.outcome, Completed) else None
    assert isinstance(attempt.action, ImportRun) and isinstance(result, ImportedRunResult)
    job = attempt.action.harbor.job_name
    series = re.search(r"-(\d{3})$", job)
    end = result.end
    ending = (
        "answered"
        if isinstance(end, ImportedSubmitted)
        else end.reason
        if isinstance(end, ImportedUnsuccessful)
        else f"invalid: {end.cause}"
    )
    return {
        "job": job,
        "trial_id": attempt.action.harbor.trial_id,
        "model": result.agent.model_id,
        "harness": result.agent.harness,
        "harness_version": result.agent.harness_version,
        "series": series.group(1) if series else None,
        "ending": ending,
        "wall_s": _r(result.wall_seconds, 1),
        "tool_calls": len(result.trajectory),
        "cost_usd": result.cost_usd,
        "budget_wall_seconds": result.budget_wall_seconds,
    }


def _custody_dir() -> Path | None:
    configured = os.environ.get("AXA_WORKBENCH_CUSTODY")
    return Path(configured) if configured else None


def _reference(case_id: str) -> dict[str, Any] | None:
    """The custody reference phase list of a case (custody.json reference_phase_coverage), read
    only by the reveal route after every assigned rater signed off (KTD12)."""
    custody = _custody_dir()
    path = custody / case_id / "custody.json" if custody else None
    if path is None or not path.is_file():
        return None
    record = json.loads(path.read_bytes())
    return {
        "source": "custody reference_phase_coverage",
        "phases": [
            {
                "formula": p["formula"],
                "space_group": p.get("space_group"),
                "weight_percent": p.get("weight_percent"),
                "candidates": p.get("shipped_candidates", []),
            }
            for p in record.get("reference_phase_coverage", [])
        ],
    }


def _match(
    phases: Sequence[str], reference: Mapping[str, Any], basis: Literal["answer", "best_fit"]
) -> dict[str, Any]:
    """Formula-level: a reference phase is found when the phases name one of its shipped
    candidates. Every polymorph of the formula counts, so this is not a polymorph check.
    `basis` says whether the phases are the answer or, with no answer, the best fit."""
    covered = {cid: p["formula"] for p in reference["phases"] for cid in p["candidates"]}
    found = {covered[c] for c in phases if c in covered}
    missed = [p["formula"] for p in reference["phases"] if p["formula"] not in found]
    extra = [c for c in phases if c not in covered]
    return {
        "basis": basis,
        "rule": "formula",
        "phase_list_exact": not missed and not extra,
        "missed": missed,
        "extra": extra,
    }
