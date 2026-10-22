"""Chemistry evidence per answered phase: charge balance, hull distance, element balance (N3).

Target path: src/phaseeval/workbench/chemistry.py (public). Overnight contract commit (AXA-43zwk.20);
the U5 worker filled the bodies and widened the types where the science review asked.

Shown as evidence beside "plausible for this sample", never as a gate (plan KTD5; the July
CHGNet hull re-gate was killed as a gate, research/energy_falsification/RESULTS.md). Every value
names its source; an unknown stays unknown.

- Charge balance is a formal oxidation-state assignment from pymatgen guesses, not a measured
  valence and not phase validation. "no_assignment_found" means none exists under the tried
  states, never that the phase is impossible: BaO2 balances only once peroxide O(-1) is tried.
- Hull distance is looked up by the candidate CIF's own database code (COD or ICSD), never by
  formula alone, in Dara's bundled metadata. Dara matched each entry to a Materials Project
  e_above_hull by reduced formula and space group, so the number is an estimate for that
  formula and space group, not a calculation of this structure (dara data/README.md). The
  Materials Project is not queried: no key is configured on the rig.
- Element balance compares the answer with the declared precursors only. An element outside
  them is a question for the rater (C and O exchange with air), not a closed-system violation.
  Missing or unreadable precursor context is not_computable, never an empty declared set.
- A phase whose formula disagrees with its own CIF's formula is reported as a mismatch and
  nothing is computed on it. A Dara row listing another formula for the CIF's code gives no hull
  value (on the demo cases: COD 4031381 La(OH)3 listed as LaO3, COD 1514093 Li2O as Li7.84O4).
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Sequence
from functools import cache
from importlib.metadata import distribution, version
from typing import Literal

from pydantic import Field
from pymatgen.core import Composition, Element

from .model import CandidateEntry, CaseManifest, Record, SafeID

Database = Literal["COD", "ICSD"]

# Dara's metadata file per database code, as dara 1.1.x bundles them.
HULL_TABLES: dict[Database, str] = {
    "COD": "cod_filtered_info_2024.json.gz",
    "ICSD": "icsd_filtered_info_2025_v3.json.gz",
}
HULL_BASIS = (
    "Materials Project e_above_hull (eV/atom) that Dara matched to this entry by reduced formula "
    "and space group, an estimate for prioritization, not a calculation of this structure "
    "(dara data/README.md)"
)
PEROXIDE_OXYGEN = {"O": [-2, -1]}
# What pymatgen raises on a formula it cannot read (an unknown symbol raises AttributeError).
PARSE_ERRORS = (ValueError, KeyError, AttributeError)


class ChargeBalance(Record):
    status: Literal["balanced", "no_assignment_found", "not_computable"]
    assignment: dict[str, float] = Field(default_factory=dict)  # element -> formal state, averaged
    alternatives: tuple[dict[str, float], ...] = ()  # further assignments, most probable first
    tried: dict[str, tuple[int, ...]] = Field(default_factory=dict)  # element -> states searched
    policy: str  # the tried-state tier that answered, or the last tier tried
    source: str


class HullMatch(Record):
    entry_id: str  # e.g. "COD 9004677"
    formula: str  # as the table writes it
    space_group_number: int
    e_above_hull_ev_per_atom: float | None


class HullDistance(Record):
    status: Literal["found", "no_value"]
    e_above_hull_ev_per_atom: float | None
    entry_id: str | None  # the candidate's own code, e.g. "COD 1530018"
    space_group_number: int | None
    source: str  # which table was checked, also when no value was found
    basis: str = HULL_BASIS
    detail: str
    # When the candidate's own code gives no value: every row of the same reduced formula in the
    # table, same space group first. Context for the rater, never this phase's value.
    formula_matches: tuple[HullMatch, ...] = ()


class PhaseInput(Record):
    phase_id: SafeID
    formula: str
    database: Database | None = None
    database_id: str | None = None  # the CIF's _cod_database_code or _database_code_ICSD
    space_group_number: int | None = None
    cif_formula: str | None = None  # the CIF's own _chemical_formula_sum


class IdentityCheck(Record):
    """Does the phase's formula agree with its own CIF's _chemical_formula_sum. On a mismatch
    nothing is computed on the phase: it is unclear which compound the evidence would describe.
    (A Dara row whose formula disagrees blocks only the hull value; see hull_distance.)"""

    status: Literal["consistent", "mismatch", "unchecked"]
    detail: str


class ElementBalance(Record):
    """The answer's elements against the declared precursors. not_computable when the
    precursor context is missing or a declared formula does not parse: an empty or partial
    declared set would flag every element, which is not a comparison (Science, AXA-43zwk.20)."""

    status: Literal["compared", "not_computable"]
    declared_elements: tuple[str, ...]  # empty when not computable
    reason: str


class PhaseChemistry(Record):
    phase_id: SafeID
    formula: str
    identity: IdentityCheck | None = None  # None: frozen in a snapshot before this check existed
    charge_balance: ChargeBalance
    hull: HullDistance
    # None when the element balance or this phase's elements could not be computed.
    elements_not_in_precursors: tuple[str, ...] | None


class ChemistryEvidence(Record):
    schema_version: Literal["chemistry-evidence.v1"] = "chemistry-evidence.v1"
    precursors: tuple[str, ...]  # declared precursor formulas, as the actor-safe brief has them
    element_balance: ElementBalance | None = None  # None: frozen in a snapshot before this status existed
    phases: tuple[PhaseChemistry, ...]
    # Elements the precursors declare that no answered phase contains (a hint, not a gate).
    # None when the element balance or any answered phase's elements could not be computed.
    precursor_elements_unaccounted: tuple[str, ...] | None


def chemistry_evidence(phases: Sequence[PhaseInput], precursors: Sequence[str]) -> ChemistryEvidence:
    """Pure apart from reading Dara's bundled metadata; no network call."""
    balance = element_balance(precursors)
    declared = set(balance.declared_elements)
    results = []
    for phase in phases:
        identity = identity_check(phase)
        elements = _elements(phase.formula) if identity.status != "mismatch" else None
        if identity.status == "mismatch":
            reason = f"not computed: {identity.detail}"
            source = f"pymatgen {version('pymatgen')} Composition.oxi_state_guesses"
            cb = ChargeBalance(status="not_computable", policy=reason, source=source)
            hull = HullDistance(
                status="no_value", e_above_hull_ev_per_atom=None,
                entry_id=f"{phase.database} {phase.database_id}" if phase.database_id else None,
                space_group_number=phase.space_group_number, source="not looked up", detail=reason,
            )
        else:
            cb, hull = charge_balance(phase.formula), hull_distance(phase)
        compared = balance.status == "compared" and elements is not None
        results.append(
            PhaseChemistry(
                phase_id=phase.phase_id, formula=phase.formula, identity=identity, charge_balance=cb,
                hull=hull, elements_not_in_precursors=tuple(sorted(elements - declared)) if compared else None,
            )
        )
    known = [r.elements_not_in_precursors is not None for r in results]
    answered = set().union(*(_elements(r.formula) or set() for r in results))
    return ChemistryEvidence(
        precursors=tuple(precursors),
        element_balance=balance,
        phases=tuple(results),
        precursor_elements_unaccounted=tuple(sorted(declared - answered)) if all(known) and balance.status == "compared" else None,
    )


def element_balance(precursors: Sequence[str]) -> ElementBalance:
    if not precursors:
        return ElementBalance(status="not_computable", declared_elements=(), reason="the case declares no precursors")
    unreadable = [p for p in precursors if not _elements(p)]
    if unreadable:
        listed = ", ".join(repr(p) for p in unreadable)
        return ElementBalance(
            status="not_computable", declared_elements=(), reason=f"declared precursor {listed} is not a readable formula"
        )
    return ElementBalance(
        status="compared",
        declared_elements=tuple(sorted(set().union(*(_elements(p) for p in precursors)))),
        reason="declared precursors only, not a closed-system check: C and O can exchange with air",
    )


def identity_check(phase: PhaseInput) -> IdentityCheck:
    if phase.cif_formula is None:
        return IdentityCheck(status="unchecked", detail="no CIF formula to compare")
    if not _same_composition(phase.formula, phase.cif_formula):
        return IdentityCheck(
            status="mismatch", detail=f"formula {phase.formula} disagrees with its CIF formula {phase.cif_formula}"
        )
    return IdentityCheck(status="consistent", detail=f"formula agrees with its CIF formula {phase.cif_formula}")


def candidate_phase(candidate: CandidateEntry, cif_text: str) -> PhaseInput:
    """The chemistry input for one case-library candidate, its code, space group and formula read
    from its own CIF (the same tags scripts/workbench/package_cases.py reads)."""
    database, database_id = None, None
    if match := re.search(r"^_cod_database_code\s+(\d+)", cif_text, re.M):
        database, database_id = "COD", match.group(1)
    elif match := re.search(r"^_database_code_ICSD\s+(\d+)", cif_text, re.M):
        database, database_id = "ICSD", match.group(1)
    sg = re.search(r"^_(?:space_group_IT_number|symmetry_Int_Tables_number)\s+(\d+)", cif_text, re.M)
    formula = re.search(r"^_chemical_formula_sum[ \t]+(?:'([^']*)'|\"([^\"]*)\"|(\S.*?))[ \t]*$", cif_text, re.M)
    return PhaseInput(
        phase_id=candidate.candidate_id,
        formula=candidate.formula_sum,
        database=database,
        database_id=database_id,
        space_group_number=int(sg.group(1)) if sg else None,
        cif_formula=next((g for g in formula.groups() if g is not None), None) if formula else None,
    )


def case_precursors(case: CaseManifest) -> tuple[str, ...]:
    return tuple(p.formula.value for p in case.material.precursors)


def charge_balance(formula: str) -> ChargeBalance:
    """pymatgen's ICSD oxidation states first, then the same with peroxide O(-1)."""
    source = f"pymatgen {version('pymatgen')} Composition.oxi_state_guesses"
    try:
        composition = Composition(formula)
        policy = "pymatgen ICSD oxidation states"
        if not all(float(n).is_integer() for n in composition.values()):
            composition = Composition(composition.get_integer_formula_and_factor()[0])
            policy += f", on the integer formula {composition.formula.replace(' ', '')}"
        tiers = [(policy, {})]
        if "O" in composition:
            tiers.append((f"{policy} plus peroxide O(-1)", PEROXIDE_OXYGEN))
        for policy, override in tiers:
            tried = {el.symbol: _states(el, override) for el in composition.elements}
            guesses = composition.oxi_state_guesses(oxi_states_override=override or None)
            if guesses:
                first, *rest = (dict(g) for g in guesses)
                return ChargeBalance(
                    status="balanced", assignment=first, alternatives=tuple(rest[:4]),
                    tried=tried, policy=policy, source=source,
                )
    except PARSE_ERRORS as error:
        return ChargeBalance(status="not_computable", policy=f"not computed: {error}", source=source)
    return ChargeBalance(status="no_assignment_found", tried=tried, policy=policy, source=source)


def hull_distance(phase: PhaseInput) -> HullDistance:
    """The table row of the candidate's own code; formula matches are listed only beside a
    missing value, never in its place."""
    database = phase.database or "COD"
    table = HULL_TABLES[database]
    source = f"dara {version('dara-xrd')} data/{table}"
    row = _hull_index(database).get(phase.database_id or "")
    if row is not None and not _same_composition(phase.formula, row[0]):
        return HullDistance(
            status="no_value", e_above_hull_ev_per_atom=None, entry_id=f"{database} {row[1]}",
            space_group_number=phase.space_group_number, source=source,
            detail=f"Dara lists {row[0]} for {database} {row[1]}, not {phase.formula}: its value would describe another composition",
            formula_matches=_formula_matches(database, phase),
        )
    if row is not None and row[3] is not None:
        return HullDistance(
            status="found", e_above_hull_ev_per_atom=row[3], entry_id=f"{database} {row[1]}",
            space_group_number=row[2], source=source, detail=f"row for {database} {row[1]}",
        )
    if phase.database is None:
        detail = "the candidate CIF carries no COD or ICSD code; no row looked up by code"
    elif row is None:
        detail = f"no row for {database} {phase.database_id}"
    else:
        detail = f"row for {database} {phase.database_id} has no e_above_hull"
    return HullDistance(
        status="no_value", e_above_hull_ev_per_atom=None,
        entry_id=f"{database} {phase.database_id}" if phase.database_id else None,
        space_group_number=phase.space_group_number, source=source, detail=detail,
        formula_matches=_formula_matches(database, phase),
    )


def _formula_matches(database: Database, phase: PhaseInput) -> tuple[HullMatch, ...]:
    try:
        composition = Composition(phase.formula)
    except PARSE_ERRORS:
        return ()
    rows = _hull_table(database).get(composition.chemical_system, [])
    matches = [
        HullMatch(entry_id=f"{database} {r[1]}", formula=r[0], space_group_number=r[2], e_above_hull_ev_per_atom=r[3])
        for r in rows
        if _reduced(r[0]) == composition.reduced_formula
    ]
    return tuple(sorted(matches, key=lambda m: m.space_group_number != phase.space_group_number))


def _states(element: Element, override: dict[str, list[int]]) -> tuple[int, ...]:
    # The list pymatgen searches when all_oxi_states is False (Composition.oxi_state_guesses).
    if element.symbol in override:
        return tuple(override[element.symbol])
    return tuple(element.icsd_oxidation_states or element.common_oxidation_states)


def _elements(formula: str) -> set[str] | None:
    """None when pymatgen cannot read the formula (an empty string reads as no elements)."""
    try:
        elements = Composition(formula).elements
    except PARSE_ERRORS:
        return None
    # pymatgen reads an unknown symbol such as "Qq" as a dummy species, not an error.
    if not elements or not all(isinstance(e, Element) for e in elements):
        return None
    return {e.symbol for e in elements}


def _same_composition(a: str, b: str) -> bool:
    try:
        return Composition(a).fractional_composition.almost_equals(Composition(b).fractional_composition, rtol=1e-3)
    except PARSE_ERRORS:
        return False


def _reduced(formula: str) -> str | None:
    try:
        return Composition(formula).reduced_formula
    except PARSE_ERRORS:
        return None


@cache
def _hull_table(database: Database) -> dict[str, list[list]]:
    """Chemical system -> rows of (formula, id, space group number, e_above_hull or None)."""
    # Located through the dara-xrd install record, not by importing dara: the launcher's
    # environment installs dara-xrd for these tables only, without its refinement stack.
    path = distribution("dara-xrd").locate_file(f"dara/data/{HULL_TABLES[database]}")
    with gzip.open(path) as handle:
        return json.load(handle)


@cache
def _hull_index(database: Database) -> dict[str, list]:
    return {row[1]: row for rows in _hull_table(database).values() for row in rows}
