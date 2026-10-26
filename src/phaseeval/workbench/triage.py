"""Triage (C1a): route only the analyses that need an expert, and keep every routing decision so
it can be measured later against that expert's sign-off.

Callers: the lab's batch script, the dashboard, qualification.Qualifier (which back-tests any
CheckPolicy against past sign-offs), and snapshot.py in the view lane, which reads its K
thresholds from PROVISIONAL_2026_10_21. So this module imports neither snapshot nor labels at
import time.

Invariants
- A check is evidence measured on a frozen AnalysisSnapshot. A routing decision is that evidence
  read through one versioned CheckPolicy. Snapshot v1 still freezes statuses, so every
  SnapshotCheck cites the policy_version it was computed under and `route` refuses a reading
  computed under another version (PolicyMismatch).
- `route` is pure. The same (readings, policy) always gives the same RouteDecision bytes, so the
  decision is published once at routing/<snapshot>/<policy.version>.json and a rerun is a no-op.
  The first write under a version also publishes the policy's own bytes, and every later write
  compares against them. A different policy under a bound version, or different decision bytes
  under an existing key, raises PolicyVersionConflict: the publish-once check, not a reviewer,
  enforces "a threshold change is a new version", even when no route changes.
- Routing is never a silent gate. A cleared analysis is still stored, exportable and reviewable.
  A deterministic audit sample of the analyses no check flagged is routed anyway, so misses have
  a denominator.
- Fail closed. A routing check that cannot be computed, or has no reading, routes to an expert.
  This is fixed behavior, not a policy option.
- Advisory checks never route. A check outside `routes_on` is shown and kept in the decision; it
  cannot move an analysis between routes.
- A routing check without a measured basis is marked unmeasured on every decision.
- A RouteDecision is a verdict, hidden from raters: label-mode credentials are refused every read
  here but one, and only a principal who may export the root reads the ledger. A rater sees a
  decision only through the revealed payload (view lane), after the store holds that rater's own
  sign-off on the snapshot and on its root's review cohort (Triage.revealed).
- A route is on the snapshot a rater signs off: the sweep routes an answer's frozen agent_answer
  snapshot when one exists and freezes one only when none does (KTD10).

Store layout (under the WorkbenchStore root):

    routing-policies/<policy_version>.json        CheckPolicy, published once: binds the version
    routing/<snapshot>/<policy_version>.json      RouteDecision, published once
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from phaseeval.api.auth import Access, Op, Principal

from .model import (
    AttemptRef,
    ChildAttempt,
    Completed,
    Digest,
    ImportedRunResult,
    ImportedSubmitted,
    ImportRun,
    Record,
    SafeID,
    canonical_bytes,
)
from .store import AttemptAccessError, CustodyError, WorkbenchStore
from phaseeval.kernel.errors import Conflict, Denied
from phaseeval.kernel.ledger import KeyShape, Mode

if TYPE_CHECKING:
    from .labels import LabelStore, SignOff
    from .model import Attempt
    from .snapshot import AnalysisSnapshot

CheckCode = Literal["K1", "K2", "K3", "K5", "BGMN_REVIEW"]
CheckStatus = Literal["pass", "flag", "not_computable"]
Route = Literal["expert_flagged", "expert_audit", "cleared"]

CHECK_ORDER: tuple[CheckCode, ...] = ("K1", "K2", "K3", "K5", "BGMN_REVIEW")
ROUTES: tuple[Route, ...] = ("expert_flagged", "expert_audit", "cleared")
ROUTING_RULE = "route-v1"  # the logic in `route`; changing it is a new CheckPolicy.version too

_DIGEST: TypeAdapter[str] = TypeAdapter(Digest)
_VERSION: TypeAdapter[str] = TypeAdapter(SafeID)


class MeasuredOn(Record):
    """What a threshold was measured on. Printed beside every number it yields.

    The counts are distinct recorded fits labeled by comparing each fit's phase list with the
    custody reference list, not expert judgments. A matched fit has the reference list exactly. A
    near-miss fit omits a reference phase or has a wrong polymorph or structure, at final Rwp <=
    15. Fits of one scan are not independent. `scans` is the distinct scan count; independence
    of the scans is not established (repeat scans can share material)."""

    source: str = Field(min_length=1)  # document path and commit
    scans: int = Field(ge=1)
    instruments: int = Field(ge=1)
    matched_fits: int = Field(ge=0)
    matched_flagged: int = Field(ge=0)
    near_miss_fits: int = Field(ge=0)
    near_miss_flagged: int = Field(ge=0)


class CheckPolicy(Record):
    """Versioned, provisional routing thresholds (the evidence.ReplayBounds pattern). No field has
    a default: a threshold nobody justified cannot be constructed. Every threshold a v1 status was
    computed under lives here, routing or not; `routes_on` alone decides which checks route. Any
    field change, or a change to ROUTING_RULE, is a new `version`."""

    version: SafeID
    provisional: bool  # True: rendered as "provisional" wherever a decision or report shows
    justified_by: str = Field(min_length=1)  # the document holding the measured basis
    routes_on: tuple[CheckCode, ...] = Field(min_length=1)  # sorted; the rest are advisory
    # Every routing check names its basis. "unmeasured" is allowed and printed on each decision.
    basis: Mapping[CheckCode, MeasuredOn | Literal["unmeasured"]]
    # K1: Rietveld degraded after a good Pawley fit.
    k1_pawley_good_max_rwp_percent: float = Field(gt=0)
    k1_rise_flag_points: float = Field(gt=0)
    # K2: unexplained residual peak height as % of the strongest net observed peak.
    k2_flag_height_percent: float = Field(gt=0)
    k2_list_height_percent: float = Field(gt=0)  # listed as a weak peak below the flag height
    # K3: largest a, b or c shift from the CIF cell, on phases at or above a weight.
    k3_flag_shift_percent: float = Field(gt=0)
    k3_min_weight_percent: float = Field(ge=0)
    # K5: reported amount minus cited-fit amount, wt% points.
    k5_flag_diff_wt_pct: float = Field(gt=0)
    # Share of unflagged analyses routed anyway, chosen by a seeded hash of the snapshot digest.
    audit_rate: float = Field(ge=0, le=1)
    audit_seed: SafeID

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.routes_on != tuple(sorted(set(self.routes_on))):
            raise ValueError("routes_on is sorted and names each check once")
        missing = set(self.routes_on) - set(self.basis)
        if missing:
            raise ValueError(f"routing checks without a basis: {sorted(missing)}")
        if self.k2_list_height_percent > self.k2_flag_height_percent:
            raise ValueError("a K2 peak is listed before it is flagged")
        return self

    def routing_checks(self) -> frozenset[CheckCode]:
        return frozenset(self.routes_on)

    @property
    def sha256(self) -> Digest:
        """Binds a version to its exact thresholds, basis and audit rule."""
        return hashlib.sha256(canonical_bytes(self)).hexdigest()


_CHECKPOINT_CHECKS = "docs/research/2026-10-21-eval-direction/checkpoint-checks.md@828c9a13b"


def _measured(matched_flagged: int, near_miss_flagged: int) -> MeasuredOn:
    # checkpoint-checks.md: 33 distinct fits matching the reference phase list and 25 near-miss
    # fits, from 7 scans on one instrument (Aeris, PIXcel/Medipix, Cu K-alpha) and one staged BGMN
    # recipe. The near-miss fits come from 5 of those scans.
    return MeasuredOn(
        source=_CHECKPOINT_CHECKS,
        scans=7,
        instruments=1,
        matched_fits=33,
        matched_flagged=matched_flagged,
        near_miss_fits=25,
        near_miss_flagged=near_miss_flagged,
    )


# Copied from the K rows of checkpoint-checks.md (research/checkpoint-checks, 828c9a13b).
PROVISIONAL_2026_10_21 = CheckPolicy(
    version="kchecks-2026-10-21-p1",
    provisional=True,
    justified_by=(
        "docs/research/2026-10-21-eval-direction/checkpoint-checks.md on branch "
        "research/checkpoint-checks (828c9a13b)"
    ),
    # K2 is the only check that separates near-miss fits from reference-matching ones. K5 catches
    # an answer that disagrees with its own cited fit. K1 never fired without Rwp > 15 and K3 is
    # recommended only together with K2, so both stay advisory.
    routes_on=("K2", "K5"),
    basis={
        "K1": _measured(matched_flagged=0, near_miss_flagged=0),  # Pawley <= 15, rise >= 10
        "K2": _measured(matched_flagged=1, near_miss_flagged=21),  # unexplained residual >= 5%
        "K3": _measured(matched_flagged=3, near_miss_flagged=9),  # an axis >= 1%, phase >= 2 wt%
        "K5": "unmeasured",  # 0/10 recorded answers flagged; no wrong answer recorded
    },
    k1_pawley_good_max_rwp_percent=15.0,
    k1_rise_flag_points=10.0,
    k2_flag_height_percent=5.0,
    k2_list_height_percent=2.0,  # 2 to 5% holds the disputed DEV006 minor phase: shown, no flag
    k3_flag_shift_percent=1.0,
    k3_min_weight_percent=2.0,
    # The doc measured answers within 0.004 points of their fit; 0.01 is the view lane's tolerance
    # for answers rounded to 4 decimals.
    k5_flag_diff_wt_pct=0.01,
    audit_rate=0.10,  # pending owner Q5
    audit_seed="kchecks-2026-10-21-p1-audit",
)


class CheckReading(Record):
    """One check as `route` reads it: from snapshot v1 statuses (readings_v1) tonight, from
    threshold-free measurements after snapshot v2. `route` does not know which."""

    code: CheckCode
    status: CheckStatus
    value: float | None  # the measured quantity in the threshold's unit; None when not frozen
    computed_under: SafeID | None  # v1: SnapshotCheck.policy_version. v2: None (route thresholds)


def readings_v1(snapshot: AnalysisSnapshot) -> tuple[CheckReading, ...]:
    """Adapter over v1 SnapshotCheck statuses. Deleted with snapshot v2, when readings come from
    measurements under the policy's thresholds. v1 freezes a value only for K2: the height of
    the highest listed residual peak (None when no peak reaches the listing height)."""
    highest = max((p.height_percent for p in snapshot.residual_peaks), default=None)
    return tuple(
        CheckReading(
            code=check.code,
            status=check.status,
            value=highest if check.code == "K2" and check.status != "not_computable" else None,
            computed_under=check.policy_version,
        )
        for check in snapshot.checks
    )


class CheckOutcome(Record):
    code: CheckCode
    status: CheckStatus
    routes: bool  # False for an advisory check: never moves the route
    value: float | None
    threshold: float | None  # the policy field that decided; None when advisory or not computable
    unmeasured: bool  # True when the check routes on a threshold with basis "unmeasured"


class RouteDecision(Record):
    """Stored once per (snapshot, policy version). Joined to labels.SignOff on `snapshot` by
    grade_routing."""

    schema_version: Literal["route.v1"] = "route.v1"
    snapshot: Digest
    root: AttemptRef
    policy_version: SafeID
    policy_sha256: Digest  # the exact policy bytes, so one version cannot name two policies
    rule: Literal["route-v1"] = ROUTING_RULE
    provisional: bool
    checks: tuple[CheckOutcome, ...]
    route: Route

    @model_validator(mode="after")
    def flagged_iff_a_routing_check_did_not_pass(self) -> Self:
        if (self.route == "expert_flagged") != any(
            c.routes and c.status != "pass" for c in self.checks
        ):
            raise ValueError("an analysis is flagged exactly when a routing check did not pass")
        return self

    @property
    def unmeasured_flag(self) -> bool:
        """Flagged only by checks whose basis is "unmeasured"."""
        causes = [c for c in self.checks if c.routes and c.status != "pass"]
        return bool(causes) and all(c.unmeasured for c in causes)


class TriageQueue(Record):
    policy_version: SafeID
    provisional: bool
    routed: tuple[RouteDecision, ...]  # expert_flagged, then expert_audit; by root and snapshot
    counts: Mapping[Route, int]
    not_computable: int  # analyses with at least one routing check not computed (all routed)
    unmeasured_flags: int  # analyses flagged only by a check whose basis is "unmeasured"


class PolicyVersionConflict(ValueError):
    """One policy version, two meanings: different policy bytes under a bound version, or
    different decision bytes under an existing (snapshot, version). Bump the version."""


class PolicyMismatch(ValueError):
    """A v1 reading was computed under another policy version than the one routing it."""


def _threshold(policy: CheckPolicy, code: CheckCode) -> float | None:
    match code:
        case "K1":
            return policy.k1_rise_flag_points
        case "K2":
            return policy.k2_flag_height_percent
        case "K3":
            return policy.k3_flag_shift_percent
        case "K5":
            return policy.k5_flag_diff_wt_pct
        case "BGMN_REVIEW":
            return None


def route(
    snapshot: Digest, root: AttemptRef, readings: Iterable[CheckReading], policy: CheckPolicy
) -> RouteDecision:
    """Pure: the same arguments always give the same decision bytes."""
    by_code: dict[CheckCode, CheckReading] = {}
    for reading in readings:
        if reading.computed_under not in (None, policy.version):
            raise PolicyMismatch(
                f"{reading.code} was computed under {reading.computed_under}, not {policy.version}"
            )
        if reading.code in by_code:
            raise ValueError(f"two readings of {reading.code}")
        by_code[reading.code] = reading
    outcomes = []
    for code in CHECK_ORDER:
        reading, routes = by_code.get(code), code in policy.routes_on
        if reading is None and not routes:
            continue
        status = reading.status if reading is not None else "not_computable"
        outcomes.append(
            CheckOutcome(
                code=code,
                status=status,
                routes=routes,
                value=reading.value if reading is not None else None,
                threshold=_threshold(policy, code)
                if routes and status != "not_computable"
                else None,
                unmeasured=routes and policy.basis[code] == "unmeasured",
            )
        )
    if any(o.routes and o.status != "pass" for o in outcomes):
        verdict: Route = "expert_flagged"
    else:
        verdict = "expert_audit" if audit_pick(snapshot, policy) else "cleared"
    return RouteDecision(
        snapshot=snapshot,
        root=root,
        policy_version=policy.version,
        policy_sha256=policy.sha256,
        provisional=policy.provisional,
        checks=tuple(outcomes),
        route=verdict,
    )


def audit_pick(snapshot: Digest, policy: CheckPolicy) -> bool:
    """Deterministic, so a rerun picks the same audit set, and independent of every check value,
    so the audited analyses are a uniform sample of the unflagged ones."""
    draw = int(hashlib.sha256(f"{policy.audit_seed}:{snapshot}".encode()).hexdigest(), 16)
    return draw < policy.audit_rate * 2**256


def _answered_import(attempt: Attempt) -> bool:
    """An imported trial that submitted an answer (the view lane's `answered` rule)."""
    return (
        isinstance(attempt, ChildAttempt)
        and isinstance(attempt.action, ImportRun)
        and isinstance(attempt.outcome, Completed)
        and isinstance(attempt.outcome.result, ImportedRunResult)
        and isinstance(attempt.outcome.result.end, ImportedSubmitted)
    )


def expert_queue(policy: CheckPolicy, decisions: Iterable[RouteDecision]) -> TriageQueue:
    """The expert queue of `decisions` under `policy`, as the owner's dashboard shows it."""
    decisions = tuple(decisions)
    order = {r: i for i, r in enumerate(ROUTES)}
    return TriageQueue(
        policy_version=policy.version,
        provisional=policy.provisional,
        routed=tuple(
            sorted(
                (d for d in decisions if d.route != "cleared"),
                key=lambda d: (order[d.route], d.root, d.snapshot),
            )
        ),
        counts={r: sum(d.route == r for d in decisions) for r in ROUTES},
        not_computable=sum(
            any(c.routes and c.status == "not_computable" for c in d.checks) for d in decisions
        ),
        unmeasured_flags=sum(d.unmeasured_flag for d in decisions),
    )


class Triage:
    """The only writer of routing/<snapshot>/<policy_version>.json. Shares the WorkbenchStore root
    and its publish-once primitive, like LabelStore. One public write, one public read."""

    def __init__(self, store: WorkbenchStore, labels: LabelStore, policy: CheckPolicy) -> None:
        self._store = store
        self._labels = labels
        self._policy = policy
        with store.ledger.transaction():
            if not hasattr(store, '_routing_collections'):
                store._routing_collections = (
                    store.ledger.collection('routing/{0}/{1}.json', RouteDecision, Mode.ONCE, KeyShape(('hex64', 'safe_id')), owner=__name__),
                    store.ledger.collection('routing-policies', CheckPolicy, Mode.ONCE, KeyShape(('safe_id',)), owner=__name__),
                )
        self._decisions, self._policies = store._routing_collections

    def sweep(
        self, principal: Principal, roots: tuple[AttemptRef, ...] | None = None
    ) -> TriageQueue:
        """Owner or the triage service only. For every
        answered imported trial under `roots` (all roots when None): take its frozen
        agent_answer snapshot, freezing one only when none exists (KTD10: frozen evidence is never
        regenerated, so the route is on the snapshot a rater signs off), route it under this
        Triage's policy and publish the decision once. Returns the expert queue.

        A rerun is a no-op. A policy that differs from the one already bound to its version is
        refused before anything is frozen or routed. A second version routes v1 snapshots only
        once their statuses are computed under it (PolicyMismatch otherwise); the first
        decisions stay. An unanswered episode gets no route (KTD10); the export counts it as
        unsuccessful. Legacy duplicates stay readable; the snapshot index chooses the routed answer."""

        _require_triage(principal)
        if roots is None:
            roots = tuple(r for r in self._store.root_refs() if self._may_triage(principal, r))
        else:
            for root in roots:
                self._store.access_for(principal, root).require(Op.TRIAGE)
        self._bind_policy()
        decisions = []
        for root in roots:
            for origin, attempt in self._store.walk(root):
                if not _answered_import(attempt):
                    continue
                digest = self._store.snapshots.freeze(principal, origin, "agent_answer")
                frozen = self._store.snapshots.get(digest)
                decision = route(digest, frozen.root, readings_v1(frozen), self._policy)
                self._publish(decision)
                decisions.append(decision)
        return expert_queue(self._policy, decisions)

    def decisions(self, principal: Principal, policy_version: SafeID) -> tuple[RouteDecision, ...]:
        """Every decision under `policy_version` this principal may triage. Label-mode
        credentials are refused: a rater gets a decision only through the revealed payload,
        after the rater's own sign-off on that snapshot."""
        _require_triage(principal)
        _VERSION.validate_python(policy_version)
        exportable: dict[str, bool] = {}
        found = []
        for name, version in self._decisions.keys():
            if version != policy_version:
                continue
            decision = self._decisions.get((name, policy_version))
            if decision.snapshot != name or decision.policy_version != policy_version:
                raise CustodyError("a routing record disagrees with its path")
            if decision.root not in exportable:
                exportable[decision.root] = self._may_triage(principal, decision.root)
            if exportable[decision.root]:
                found.append(decision)
        return tuple(found)

    def revealed(self, principal: Principal, snapshot: Digest) -> RouteDecision | None:
        """The one read a rater gets (snapshot.revealed_payload): this snapshot's decision under
        this Triage's policy, once the store holds the rater's own sign-off on the snapshot and on
        every snapshot of its root's review cohort. None before then, or when nothing routed it.
        The admin owner reads the ledger through `decisions`."""
        try:
            self._labels.require_identity(principal, snapshot).check(snapshot)
        except Denied:
            return None
        decision = self._decisions.get((snapshot, self._policy.version))
        if decision is None:
            return None
        if (decision.snapshot, decision.policy_sha256) != (snapshot, self._policy.sha256):
            raise CustodyError("a routing record disagrees with its path or its policy")
        return decision

    def _bind_policy(self) -> None:
        """Publish this policy's bytes once under its version; refuse any other bytes there."""
        try:
            self._policies.put((self._policy.version,), self._policy)
        except Conflict as exc:
            raise PolicyVersionConflict(f"{self._policy.version} is bound to other thresholds or audit rules: a policy change needs a new version") from exc

    def _publish(self, decision: RouteDecision) -> None:
        if (decision.policy_version, decision.policy_sha256) != (
            self._policy.version,
            self._policy.sha256,
        ):
            raise PolicyMismatch("a decision is published only under this Triage's policy")
        self._bind_policy()
        try:
            self._decisions.put((decision.snapshot, decision.policy_version), decision)
        except Conflict as exc:
            raise PolicyVersionConflict(f"snapshot {decision.snapshot} already routes differently under {decision.policy_version}: a threshold or rule change needs a new version") from exc

    def _may_triage(self, principal: Principal, root: AttemptRef) -> bool:
        try:
            self._store.access_for(principal, root).require(Op.TRIAGE)
        except (AttemptAccessError, Denied):
            return False
        return True


def _require_triage(principal: Principal) -> None:
    """A label-mode credential never reads or writes routing: its sign-off must not see the route."""
    Access.for_principal(principal).require(Op.TRIAGE)


# --- routing accuracy against sign-offs ---------------------------------------------------------

# An operational display rule, not a confidence bound: with fewer reviewed audit analyses no recall
# estimate is printed. Fits of one scan are correlated and a selectively completed audit is not a
# uniform sample, so no interval follows from this count.
RECALL_MIN_AUDIT_REVIEWS = 30


class RouteTally(Record):
    """Decisions on one route joined to the consensus of their counted sign-offs. "wrong" means
    the raters would not sign the answer, so it needed an expert. (wrong + right) / decisions is
    the route's completion; disputed and unreviewed are its unknown strata."""

    decisions: int
    wrong: int
    right: int
    disputed: int  # counted raters disagree: outside every rate
    unreviewed: int  # no counted blind sign-off: outside every rate
    reviewed_roots: int  # distinct roots among wrong and right; independence not established


class RoutingMiss(Record):
    snapshot: Digest
    root: AttemptRef
    route: Literal["expert_audit", "cleared"]


class RoutingGrade(Record):
    """Routing compared with sign-offs: descriptive counts per route, then one estimate.

    Caught = expert_flagged wrong, false alarm = expert_flagged right, missed = an unflagged
    analysis the raters would not sign. Misses found outside the audit sample (a cleared analysis
    reviewed on request) are listed but stay out of the estimate, because nothing made that sample
    uniform. Zero audited misses does not mean zero missed analyses.

    While the v1 blind page shows K pass and flag statuses, the evidence the router reads, every
    sign-off is router-exposed. This is agreement with raters who saw the checks, not an
    independent validation of triage."""

    policy_version: SafeID
    policy_sha256: Digest
    provisional: bool
    audit_rate: float  # each unflagged analysis's selection probability into the audit sample
    rater_kinds: tuple[str, ...]  # whose sign-offs were counted, e.g. ("owner",)
    by_route: Mapping[Route, RouteTally]
    misses: tuple[RoutingMiss, ...]
    unmeasured_flags: int
    # Descriptive, not a measured population recall: flagged-wrong over flagged-wrong plus
    # unflagged-wrong, each extrapolated from the reviewed decisions of its stratum. It assumes
    # the reviewed decisions of a stratum represent the unreviewed ones and the audited analyses
    # represent the cleared ones. One audited miss stands for 1 / audit_rate unflagged analyses,
    # so an ordinary binomial interval does not apply. None below RECALL_MIN_AUDIT_REVIEWS.
    recall_estimate: float | None
    recall_withheld: str | None

    @property
    def caught(self) -> int:
        return self.by_route["expert_flagged"].wrong

    @property
    def false_alarms(self) -> int:
        return self.by_route["expert_flagged"].right

    @property
    def missed(self) -> int:
        return len(self.misses)


def _consensus(
    signoffs: Iterable[SignOff],
) -> tuple[dict[str, Literal["wrong", "right", "disputed"]], tuple[str, ...]]:
    """Each rater's latest blind verdict per snapshot, and the rater kinds counted. A revealed
    sign-off may be anchored on the reveal; an unresolved one judges nothing."""
    latest: dict[tuple[str, str], SignOff] = {}
    for signoff in signoffs:
        if signoff.rater.kind == "model_judge":
            raise ValueError("a model judge's sign-off never grades routing")
        if signoff.exposure != "blind":
            continue
        key = (signoff.snapshot, signoff.rater.principal_id)
        if key not in latest or signoff.created_at_utc > latest[key].created_at_utc:
            latest[key] = signoff
    counted = [s for s in latest.values() if s.verdict != "unresolved"]
    verdicts: dict[str, set[str]] = {}
    for signoff in counted:
        verdicts.setdefault(signoff.snapshot, set()).add(signoff.verdict)
    calls = {
        snapshot: "disputed" if len(v) > 1 else ("wrong" if v == {"would_not_sign"} else "right")
        for snapshot, v in verdicts.items()
    }
    return calls, tuple(sorted({s.rater.kind for s in counted}))


def grade_routing(
    decisions: Iterable[RouteDecision],
    signoffs: Iterable[SignOff],
    policy: CheckPolicy,
    *,
    min_audit_reviews: int,
) -> RoutingGrade:
    """Pure. Join on snapshot digest. The caller chooses whose sign-offs count (a lab's experts,
    or the owner tonight) and passes only those; a model judge's sign-off is refused."""
    decisions = tuple(decisions)
    if min_audit_reviews < 1:
        raise ValueError("the estimate needs at least one reviewed audit analysis")
    if any(
        (d.policy_version, d.policy_sha256) != (policy.version, policy.sha256) for d in decisions
    ):
        raise PolicyMismatch(f"grade decisions under one policy at a time: {policy.version}")
    if len({d.snapshot for d in decisions}) != len(decisions):
        raise ValueError("one decision per snapshot")
    consensus, rater_kinds = _consensus(signoffs)
    tallies, misses = {}, []
    for r in ROUTES:
        on_route = [d for d in decisions if d.route == r]
        calls = [consensus.get(d.snapshot) for d in on_route]
        tallies[r] = RouteTally(
            decisions=len(on_route),
            wrong=calls.count("wrong"),
            right=calls.count("right"),
            disputed=calls.count("disputed"),
            unreviewed=calls.count(None),
            reviewed_roots=len(
                {
                    d.root
                    for d, call in zip(on_route, calls, strict=True)
                    if call in ("wrong", "right")
                }
            ),
        )
        if r != "expert_flagged":
            misses += [
                RoutingMiss(snapshot=d.snapshot, root=d.root, route=r)
                for d, call in zip(on_route, calls, strict=True)
                if call == "wrong"
            ]
    flagged, audited, cleared = (tallies[r] for r in ROUTES)
    audit_reviews = audited.wrong + audited.right
    flagged_reviews = flagged.wrong + flagged.right
    recall, withheld = None, None
    if audit_reviews < min_audit_reviews:
        withheld = (
            f"{audit_reviews} reviewed audit analyses; the estimate needs {min_audit_reviews}"
        )
    elif flagged.decisions and not flagged_reviews:
        withheld = "no flagged analysis has a counted sign-off"
    else:
        flagged_wrong = (
            flagged.wrong / flagged_reviews * flagged.decisions if flagged_reviews else 0.0
        )
        unflagged_wrong = audited.wrong / audit_reviews * (audited.decisions + cleared.decisions)
        if flagged_wrong + unflagged_wrong:
            recall = flagged_wrong / (flagged_wrong + unflagged_wrong)
        else:
            withheld = "no reviewed analysis was wrong"
    return RoutingGrade(
        policy_version=policy.version,
        policy_sha256=policy.sha256,
        provisional=policy.provisional,
        audit_rate=policy.audit_rate,
        rater_kinds=rater_kinds,
        by_route=tallies,
        misses=tuple(sorted(misses, key=lambda m: (m.root, m.snapshot))),
        unmeasured_flags=sum(d.unmeasured_flag for d in decisions),
        recall_estimate=recall,
        recall_withheld=withheld,
    )


__all__ = [
    "CHECK_ORDER",
    "CheckCode",
    "CheckOutcome",
    "CheckPolicy",
    "CheckReading",
    "MeasuredOn",
    "PROVISIONAL_2026_10_21",
    "PolicyMismatch",
    "PolicyVersionConflict",
    "RECALL_MIN_AUDIT_REVIEWS",
    "ROUTES",
    "Route",
    "RouteDecision",
    "RouteTally",
    "RoutingGrade",
    "RoutingMiss",
    "Triage",
    "TriageQueue",
    "audit_pick",
    "expert_queue",
    "grade_routing",
    "route",
]
