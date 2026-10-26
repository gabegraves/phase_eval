import hashlib
import json
from typing import Annotated

import pytest

from phaseeval.kernel.errors import AudienceViolation, Denied
from phaseeval.kernel.records import Audience, canonical_bytes, check_audiences
from phaseeval.workbench.judge_inputs import BlindCase, blind_view, judge_input_bytes, public_blind_view
from phaseeval.workbench.labels import IdentityProof, JudgeRunManifest, ReferenceProof
from tests.support.store import RATER_A, RATER_B


def test_frozen_public_projection_and_exact_judge_bytes(assigned):
    frozen = assigned.labels.snapshot(RATER_A, assigned.snaps[0])
    view = public_blind_view(blind_view(frozen, item='h' + 'a' * 26,
                                       case=BlindCase(task='identify phases', context={})))
    raw = judge_input_bytes(view)
    assert raw == canonical_bytes(view)
    assert raw == (json.dumps(view.model_dump(mode='json'), sort_keys=True, separators=(',', ':')) + '\n').encode()
    assert view.answer.phases[0].phase_id == frozen.phases[0].phase_id
    assert view.checks[0].status == frozen.checks[0].status
    manifest = JudgeRunManifest(run_id='run', model='judge', prompt_sha256='a' * 64,
                                input_scope='dataset:release', input_sha256={view.item: hashlib.sha256(raw).hexdigest()})
    assert manifest.input_sha256[view.item] == hashlib.sha256(raw).hexdigest()
    check_audiences(view, frozenset({Audience.PUBLIC}))
    check_audiences(view, frozenset({Audience.RATER_BLIND}))
    assert judge_input_bytes(type(view).model_validate_json(raw)) == raw

    class PrivateCase(BlindCase):
        secret: Annotated[str, Audience.EXPERT]

    leaked = view.model_copy(update={'case': PrivateCase(task='task', context={}, secret='model')})
    with pytest.raises(AudienceViolation, match='case.secret'):
        judge_input_bytes(leaked)


def test_identity_and_reference_proofs_are_bound_to_their_target(assigned):
    snap = assigned.snaps[0]
    for kind, key in ((IdentityProof, {'snapshot': snap}), (ReferenceProof, {'family': 'a' * 64})):
        with pytest.raises(TypeError):
            kind(principal_id='rater-a', **key)
    with pytest.raises(Denied):
        assigned.labels.require_identity(RATER_A, snap)
    assigned.sign_all(RATER_A)
    proof = assigned.labels.require_identity(RATER_A, snap)
    proof.check(snap)
    with pytest.raises(Denied):
        proof.check(assigned.snaps[1])
    with pytest.raises(Denied):
        assigned.labels.require_reference(RATER_A, assigned.root)
    assigned.sign_all(RATER_B)
    reference = assigned.labels.require_reference(RATER_A, assigned.root)
    reference.check(assigned.labels.snapshot(RATER_A, snap).pattern_sha256)
    with pytest.raises(Denied):
        reference.check('b' * 64)
