from concurrent.futures import ThreadPoolExecutor

import pytest

from phaseeval.api.auth import CaseOperation
from phaseeval.workbench.labels import SignOffRequest
from phaseeval.workbench.review import NoteRequest, PreferenceRequest
from phaseeval.workbench.store import ActionRequest
from tests.support.store import ADD, OWNER, RATER_A, REFINE


@pytest.mark.parametrize('operation', ['attempt', 'note', 'preference'])
def test_tree_publication_uses_exposure_after_a_final_signoff(assigned, monkeypatch, operation):
    store = assigned.store
    store.grant(OWNER, assigned.root, RATER_A.principal_id, (CaseOperation.ACTION,))
    if operation == 'preference':
        chosen, rejected = (store.act(RATER_A, assigned.root, ActionRequest(
            idempotency_key=key, action=ADD)).digest for key in ('chosen', 'rejected'))
    assigned.labels.sign_off(RATER_A, SignOffRequest(snapshot=assigned.snaps[0], verdict='would_sign'))
    clock = store._clock
    signed = False

    def sign_before_publication():
        nonlocal signed
        if not signed:
            signed = True
            assigned.labels.sign_off(RATER_A, SignOffRequest(snapshot=assigned.snaps[1], verdict='would_sign'))
        return clock()

    monkeypatch.setattr(store, '_clock', sign_before_publication)
    if operation == 'attempt':
        request = ActionRequest(idempotency_key='across-signoff', action=ADD)
        result = store.act(RATER_A, assigned.root, request)
        published = dict(store.walk(assigned.root))[result.digest]
        actor = published.actor
        assert store.act(RATER_A, assigned.root, request).digest == result.digest
    else:
        target = chosen if operation == 'preference' else assigned.root
        request = (PreferenceRequest(chosen=chosen, rejected=rejected, reason='across signoff')
                   if operation == 'preference' else NoteRequest(text='across signoff'))
        result = store.record(RATER_A, target, request)
        published = store.annotations(target)[-1]
        assert result == published
        actor = published.author
    assert signed
    assert assigned.labels.reveal_state(RATER_A, assigned.root).signed_off
    assert actor.reference_exposure == 'exposed'


@pytest.mark.parametrize('operation', ['attempt', 'annotation'])
def test_a_final_signoff_cannot_interleave_with_tree_publication(assigned, monkeypatch, operation):
    store = assigned.store
    store.grant(OWNER, assigned.root, RATER_A.principal_id, (CaseOperation.ACTION,))
    assigned.labels.sign_off(RATER_A, SignOffRequest(snapshot=assigned.snaps[0], verdict='would_sign'))
    signoff = SignOffRequest(snapshot=assigned.snaps[1], verdict='would_sign')
    signed_at_publication = []

    def try_sign():
        with store.ledger._lock.try_held() as acquired:
            if acquired:
                assigned.labels._sign_off_locked(RATER_A, signoff)
            return acquired

    method = '_put_attempt' if operation == 'attempt' else '_annotate'
    publish = getattr(store, method)

    def sign_at_publication(*args):
        with ThreadPoolExecutor(max_workers=1) as pool:
            signed_at_publication.append(pool.submit(try_sign).result(timeout=10))
        return publish(*args)

    monkeypatch.setattr(store, method, sign_at_publication)
    if operation == 'attempt':
        result = store.act(RATER_A, assigned.root, ActionRequest(idempotency_key='atomic', action=ADD))
        actor = dict(store.walk(assigned.root))[result.digest].actor
    else:
        store.record(RATER_A, assigned.root, NoteRequest(text='atomic'))
        actor = store.annotations(assigned.root)[-1].author
    assert len(signed_at_publication) == 1
    assert actor.reference_exposure == ('exposed' if signed_at_publication[0] else 'unexposed')
    if not signed_at_publication[0]:
        assigned.labels.sign_off(RATER_A, signoff)
    assert assigned.labels.reveal_state(RATER_A, assigned.root).signed_off


def test_a_refit_allows_final_signoff_during_execution(assigned, monkeypatch):
    store = assigned.store
    store.grant(OWNER, assigned.root, RATER_A.principal_id, (CaseOperation.ACTION,))
    parent = store.act(RATER_A, assigned.root, ActionRequest(idempotency_key='add', action=ADD)).digest
    assigned.labels.sign_off(RATER_A, SignOffRequest(snapshot=assigned.snaps[0], verdict='would_sign'))
    execute = store._execute

    def sign_during_execution(context, action):
        assigned.labels.sign_off(RATER_A, SignOffRequest(snapshot=assigned.snaps[1], verdict='would_sign'))
        return execute(context, action)

    monkeypatch.setattr(store, '_execute', sign_during_execution)
    result = store.act(RATER_A, parent, ActionRequest(idempotency_key='refit', action=REFINE))
    assert result.attempt.actor.reference_exposure == 'exposed'
