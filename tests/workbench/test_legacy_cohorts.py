from datetime import datetime, timedelta, timezone

from phaseeval.workbench.labels import AssignmentV1, Rater, SignOff
from phaseeval.workbench.labels import FamilyFacts, cohorts_of, identity_open, reference_open

ROOT = "f" * 64
A, B, C, D = (c * 64 for c in "abcd")
T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def assignment(at, snapshots):
    return AssignmentV1(root=ROOT, raters=("one", "two"), snapshots=snapshots,
                        assigned_by="owner", created_at_utc=T + timedelta(seconds=at))


def signs(at, who, snapshots):
    return [SignOff(root=ROOT, snapshot=s, verdict="would_sign", rater=Rater(principal_id=who, kind="expert"),
                    exposure="blind", created_at_utc=T + timedelta(seconds=at)) for s in snapshots]


def facts(history, signoffs):
    return FamilyFacts(family=ROOT, cohorts=cohorts_of(history, signoffs), signoffs=tuple(signoffs),
                       declarations=(), legacy_reveals=())


def test_three_cumulative_versions_never_reintroduce_old_snapshots():
    history = [assignment(0, (A, B)), assignment(2, (A, B, C)), assignment(4, (A, B, C, D))]
    f = facts(history, signs(1, "one", (A, B)) + signs(3, "one", (C,)))
    assert [c.snapshots for c in f.cohorts] == [(A, B), (C,), (D,)]
    assert identity_open("one", A, f)
    assert not reference_open("one", f)


def test_delayed_signoff_of_split_subset_cannot_open_reference():
    history = [assignment(0, (A, B)), assignment(2, (A, B, C))]
    f = facts(history, signs(1, "one", (A, B)) + signs(3, "two", (A, B)))
    assert all(identity_open(r, A, f) for r in ("one", "two"))
    assert not any(reference_open(r, f) for r in ("one", "two"))
    full = facts(history, list(f.signoffs) + signs(4, "one", (C,)) + signs(4, "two", (C,)))
    assert reference_open("one", full)


def test_reference_eligibility_that_existed_before_growth_is_preserved():
    f = facts([assignment(0, (A, B)), assignment(3, (A, B, C))],
              signs(1, "one", (A, B)) + signs(2, "two", (A, B)))
    assert reference_open("one", f)


def test_rater_replacement_after_signoff_preserves_identity_and_reference():
    original = assignment(0, (A,)).model_copy(update={'raters': ('one',)})
    signoffs = signs(1, 'one', (A,))
    before = facts([original], signoffs)
    replacement = original.model_copy(update={'raters': ('two',), 'created_at_utc': T + timedelta(seconds=2)})
    after = facts([original, replacement], signoffs)
    assert identity_open('one', A, before) and reference_open('one', before)
    assert identity_open('one', A, after) and reference_open('one', after)
    assert not identity_open('two', A, after) and not reference_open('two', after)
    assert [c.raters for c in after.cohorts] == [('one',), ('two',)]


def test_rater_addition_preserves_historical_rights_without_granting_new_rater():
    original = assignment(0, (A,)).model_copy(update={'raters': ('one',)})
    addition = assignment(2, (A,))
    after = facts([original, addition], signs(1, 'one', (A,)))
    assert identity_open('one', A, after) and reference_open('one', after)
    assert not identity_open('two', A, after) and not reference_open('two', after)
    complete = facts([original, addition], signs(1, 'one', (A,)) + signs(3, 'two', (A,)))
    assert identity_open('two', A, complete) and reference_open('two', complete)
