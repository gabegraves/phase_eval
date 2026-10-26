import json
import shutil

from phaseeval.api.auth import CaseOperation
from phaseeval.workbench import cli
from phaseeval.workbench.exports import RaterPseudonyms
from phaseeval.workbench.labels import LabelStore
from tests.support.store import AGENT, OWNER, RATER_A


def test_legacy_acl_is_the_first_grant_and_is_never_rewritten(world):
    path = world.store.root / 'roots' / world.root / 'acl.json'
    data = {'schema_version': 'axa.workbench_acl.v1', 'root': world.root,
            'access': {OWNER.principal_id: ['inspect', 'action', 'export', 'artifact']}}
    original = json.dumps(data).encode()
    path.write_bytes(original)
    shutil.rmtree(path.parent / 'grants')
    world.store.grant(OWNER, world.root, AGENT.principal_id, (CaseOperation.INSPECT,))
    assert world.root in world.store.roots(AGENT)
    assert path.read_bytes() == original
    assert tuple((path.parent / 'grants').iterdir())


def test_a_legacy_acl_for_another_root_is_refused_even_after_grants(world):
    path = world.store.root / 'roots' / world.root / 'acl.json'
    path.write_text(json.dumps({'schema_version': 'axa.workbench_acl.v1', 'root': 'a'*64,
                                'access': {OWNER.principal_id: ['inspect']}}))
    assert world.root not in world.store.roots(OWNER)


def test_open_rebuilds_families_and_rater_items_from_records(assigned):
    root = assigned.store.root
    shutil.rmtree(root / 'families')
    shutil.rmtree(root / 'rater-items')
    reopened = cli.open_store(root, root / 'imports')
    labels = LabelStore(reopened)
    assert set(reopened._rater_items.members((RATER_A.principal_id,))) == set(labels.cohort(RATER_A, assigned.root))
    case = reopened.walk(assigned.root)[0][1].case
    assert assigned.root in reopened._families.members((case.pattern.sha256,))


def test_pseudonyms_keep_the_supplied_private_directory(tmp_path):
    private = tmp_path / 'identities'
    pseudonyms = RaterPseudonyms(private)
    name = pseudonyms('rater')
    written = pseudonyms.write_map()
    assert written.is_relative_to(private)
    assert written.is_file()
    assert RaterPseudonyms(private)('rater') == name
    assert not (tmp_path / 'private').exists()
