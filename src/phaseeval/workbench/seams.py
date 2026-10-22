"""The one seam between the public package and private custody.

Target path: src/phaseeval/workbench/seams.py (public). Everything in phaseeval.workbench, the workbench
page, the MCP server and the dashboard is public. What needs truth or canaries reaches the
public side only through the protocols below, injected once in `build_app`.

Candidate 2 had four protocols. The verifier and truth index are cut: adjudications arrive as
annotations from a sealed_evaluator service principal, and truth admission is a versioned
policy input, so neither needs an in-process hook at the hackathon.

Public defaults are honest, never fake guards: with no scanner, every completed outcome
records custody_scan="not_configured" and every episode stays unreviewable for training; with
NoReference, every qualification reference stance reads "absent".

The sealed reference (AXA-43zwk.26, design sketch/seams.py) is the only server-side path to a
case's reference answer. SealedReference reaches qualification.derive_expectation, which reduces
it to a per-phase stance (agrees | disputes | absent) before anything is stored; no blind
payload, judge payload, actor view, AnalysisSnapshot, RouteDecision or CaseExpectation has a
field of this type. The custody reader lives in scripts/workbench/qualify.py on the owner's host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .model import CandidateId, Digest, MaterialContext, Record, SafeID, TaskKind
from .policy import ReferencePhase


class CanaryMatch(Record):
    canary_sha256: Digest  # the scanner never returns the canary itself


class CustodyScanner(Protocol):
    """R9. Private: holds the canary planted in each case's custody copy (U2)."""

    def scan(self, data: bytes) -> CanaryMatch | None: ...


class TaskKindRule(Protocol):
    """Measurement plan R12, recorded as the R30 confirmation tag. The private rule compares
    the visible context with custody truth."""

    def task_kind(self, case_id: SafeID, material: MaterialContext) -> TaskKind: ...


class SealedReferencePhase(ReferencePhase):
    """policy.ReferencePhase (formula, space group, wt%) plus the shipped candidate ids custody
    maps it to, so a reference phase matches by candidate or by phase key."""

    candidates: tuple[CandidateId, ...]


class SealedReference(Record):
    case_id: SafeID
    source: str  # e.g. "custody reference_phase_coverage"
    phases: tuple[SealedReferencePhase, ...]
    disputed: tuple[str, ...] = ()  # formulas the owner marks disputed (DEV006 Bi2CO5)


class ReferenceReader(Protocol):
    def reference(self, case_id: SafeID) -> SealedReference | None: ...


class NoReference:
    """Public default and test reader: no case has a reference."""

    def reference(self, case_id: SafeID) -> SealedReference | None:
        return None


@dataclass(frozen=True)
class Seams:
    scanner: CustodyScanner | None
    task_kind: TaskKindRule
    reference: ReferenceReader = field(default_factory=NoReference)


class NoTruthTaskKind:
    """Public default for a lab with no truth: declared precursors and no heating step make a
    confirmation; anything that was reacted is an identification."""

    def task_kind(self, case_id: SafeID, material: MaterialContext) -> TaskKind:
        raise NotImplementedError


def public_seams() -> Seams:
    return Seams(scanner=None, task_kind=NoTruthTaskKind(), reference=NoReference())


def load_private_seams(entry_point: str) -> Seams:
    """`AXA_PRIVATE_SEAMS=axa_private.seams:build` on the owner's host. Absent publicly."""
    raise NotImplementedError  # TODO: importlib.import_module + getattr; no fallback on error
