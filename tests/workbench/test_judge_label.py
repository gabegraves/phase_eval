from __future__ import annotations

import hashlib
import json

import pytest

from phaseeval.api.auth import CaseOperation
from phaseeval.kernel.errors import Conflict, Denied, Refused
from phaseeval.workbench import labels as records
from phaseeval.workbench.judge_inputs import PublicBlindView, judge_input_bytes
from tests.support.store import JUDGE, OWNER, RATER_A


def registered(bench):
    bench.store.grant(OWNER, bench.root, JUDGE.principal_id, (CaseOperation.INSPECT,))
    manifest = bench.labels.register_judge_run(OWNER, 'judge-run', 'qwen3-32b', '8' * 64,
                                             'dataset:test', (bench.snaps[0],))
    item, digest = next(iter(manifest.input_sha256.items()))
    identity = records.JudgeIdentity(model=manifest.model, prompt_sha256=manifest.prompt_sha256,
                                    input_sha256=digest, run_id=manifest.run_id, judge_is_actor_model=False)
    return manifest, item, identity


@pytest.mark.parametrize('field,value', [('model', 'other'), ('prompt_sha256', '0' * 64),
                                         ('input_sha256', '0' * 64), ('run_id', 'other-run')])
def test_a_judge_label_with_a_mismatched_identity_is_refused(bench, field, value):
    _, item, identity = registered(bench)
    body = records.JudgeLabelBody(rubric_item='fit_supports_conclusion', answer='yes',
                                  judge=identity.model_copy(update={field: value}))
    before = bench.labels.export_records('training')[0]
    with pytest.raises(Refused):
        bench.labels.judge_label(JUDGE, item, body)
    assert bench.labels.export_records('training')[0] == before


def test_a_valid_judge_label_is_stamped_blind_with_its_bound_identity(bench):
    manifest, item, identity = registered(bench)
    data = bench.labels.judge_input(JUDGE, manifest.run_id, item)
    assert data == judge_input_bytes(PublicBlindView.model_validate_json(data))
    assert data.endswith(b'\n') and hashlib.sha256(data).hexdigest() == identity.input_sha256
    assert json.loads(data)['item'] == item
    body = records.JudgeLabelBody(rubric_item='fit_supports_conclusion', answer='yes', judge=identity)
    label = bench.labels.judge_label(JUDGE, item, body)
    assert label.snapshot == bench.snaps[0] and label.exposure == 'blind'
    assert label.rater.principal_id == JUDGE.principal_id and label.rater.kind == 'model_judge'
    assert label.rater.judge.judge_is_actor_model is True
    with pytest.raises(Denied):
        bench.labels.judge_label(RATER_A, item, body)
    with pytest.raises(Denied):
        bench.labels.register_judge_run(JUDGE, 'new', 'other', '8' * 64, 'dataset:test', (bench.snaps[0],))
    with pytest.raises(Conflict):
        bench.labels.register_judge_run(OWNER, manifest.run_id, 'other', '8' * 64, 'dataset:test', (bench.snaps[0],))
    assert bench.labels.register_judge_run(OWNER, manifest.run_id, manifest.model, '8' * 64,
                                         'dataset:test', (bench.snaps[0],)) == manifest


def test_judge_input_and_label_require_the_root_grant(bench):
    manifest, item, identity = registered(bench)
    from phaseeval.api.auth import Principal, PrincipalType
    stranger = Principal('other-judge', PrincipalType.MODEL_JUDGE, frozenset())
    with pytest.raises(Denied):
        bench.labels.judge_input(stranger, manifest.run_id, item)
    with pytest.raises(Denied):
        bench.labels.judge_label(stranger, item, records.JudgeLabelBody(
            rubric_item='fit_supports_conclusion', answer='yes', judge=identity))
