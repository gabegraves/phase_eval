import hashlib
import json

from phaseeval.workbench import cli, exports, snapshot
from phaseeval.workbench.executors import LIVE_BGMN_THREADS, recipe_digest, refinement_config
from phaseeval.workbench.scorecard import scorecard_view
from tests.support.demo import REPO


def test_the_published_dataset_validates():
    assert exports.validate_dataset(REPO / 'data/dataset-v0-public', exports.dataset_json_schemas()) == []


def test_the_demo_export_validates(demo_store):
    assert exports.validate_dataset(demo_store.directory / 'dataset', exports.dataset_json_schemas()) == []


def test_dev006_root_digests_for_both_layouts(demo_store, tmp_path):
    from phaseeval.workbench import importer
    from phaseeval.workbench.model import RootAttempt, digest_of

    harness, _ = importer.case_manifest(importer.read_case(REPO / 'harness/examples/dev006/input'))
    assert digest_of(RootAttempt(case=harness)) == 'a243f16c7356e58c6c8add0adc95bb2f60e775fc16eab3fedbd48a7cb39995f3'
    packaged = demo_store.store.walk(demo_store.roots['DEV006'])[0][1]
    assert digest_of(packaged) == 'd51a3823f766972d0fab8d133e6485b00f181d5e857456b37965ada8c8899c61'



def test_the_demo_blind_payload_key_set(demo_store, tmp_path):
    store, (origin,) = demo_store.import_into(tmp_path / 'fresh', [demo_store.origin()])
    frozen = snapshot.build_snapshot(store, cli.OWNER, origin, 'agent_answer')
    view = exports.blind_view(store, cli.OWNER, frozen, exports.digest_of(frozen))
    assert sorted(view) == [
        'answer', 'answered', 'brief', 'candidates', 'case_id', 'checks', 'chemistry', 'digest',
        'fit', 'kind', 'missing', 'origin', 'phase_window_rule', 'phase_windows', 'residual_peaks',
        'residual_windows', 'root', 'schema_version', 'screenings', 'steps', 'times',
    ]


def test_the_scorecard_view_of_the_demo_export(demo_store):
    view = scorecard_view(demo_store.directory / 'dataset', reference=lambda _: None)
    assert len(view.episodes) == 6
    assert view.reference_cases == ()
    assert all(row.ok for row in view.reconciliation)
    projection = [(e.case_id, e.model, [(s.step, s.outcome) for s in e.steps]) for e in view.episodes]
    assert hashlib.sha256(json.dumps(projection, sort_keys=True).encode()).hexdigest() == 'c30d72bd714c44bba61b1feaaa4af2c9779716cbb8cef7f74c3481ecdd92d8c3'


def test_the_recorded_agent_and_live_recipe_digests(tmp_path):
    def digest(threads):
        config = refinement_config(tmp_path, recipe='staged_rietveld', n_threads=threads,
                                   instrument_profile='Aeris-fds-Pixcel1d-Medipix3', wavelength='Cu')
        return recipe_digest(config.to_dict())

    assert digest(1) == '7ce17661e7c98fe7'
    assert digest(LIVE_BGMN_THREADS) == '267a2ad2670edc56'
