"""U10, architect tonight.md row U10: K3 never routes, for any readings under the frozen policy.

Hypothesis generates readings of every check (each present or absent, any status and value)
and a snapshot digest, and asks the real triage.route under PROVISIONAL_2026_10_21: changing
K3's reading never changes the route, and K3's outcome never routes or carries a threshold.
Skill: .claude/skills/label-blindness/SKILL.md.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

triage = pytest.importorskip("phaseeval.workbench.triage")

POLICY = triage.PROVISIONAL_2026_10_21
ROOT = "1" * 64
STATUSES = st.sampled_from(("pass", "flag", "not_computable"))
VALUES = st.none() | st.floats(min_value=0, max_value=100, allow_nan=False)


def reading(code: str, status: str, value: float | None) -> object:
    return triage.CheckReading(code=code, status=status, value=value, computed_under=POLICY.version)


@settings(max_examples=300, deadline=None, derandomize=True, database=None)
@given(
    present=st.fixed_dictionaries({code: st.booleans() for code in ("K1", "K2", "K5", "BGMN_REVIEW")}),
    statuses=st.fixed_dictionaries(dict.fromkeys(("K1", "K2", "K5", "BGMN_REVIEW"), STATUSES)),
    values=st.fixed_dictionaries(dict.fromkeys(("K1", "K2", "K5", "BGMN_REVIEW"), VALUES)),
    k3=st.none() | st.tuples(STATUSES, VALUES),
    snap=st.from_regex(r"[0-9a-f]{64}", fullmatch=True),
)
def test_k3_never_routes(present: dict, statuses: dict, values: dict, k3: tuple | None, snap: str) -> None:
    others = [reading(code, statuses[code], values[code]) for code in present if present[code]]
    with_k3 = others + ([reading("K3", *k3)] if k3 is not None else [])
    decision = triage.route(snap, ROOT, with_k3, POLICY)
    baseline = triage.route(snap, ROOT, others, POLICY)

    assert decision.route == baseline.route
    assert all(not c.routes and c.threshold is None for c in decision.checks if c.code == "K3")
