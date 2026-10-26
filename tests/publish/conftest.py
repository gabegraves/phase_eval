from datetime import UTC, datetime
from pathlib import Path

import pytest

from phaseeval.api.auth import Principal, PrincipalType
from phaseeval.workbench import exports, policy
from phaseeval.workbench.labels import StageProjection, LabelStore
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.store import WorkbenchStore


@pytest.fixture
def empty_export(tmp_path):
    store = WorkbenchStore(
        tmp_path / 'store', lambda *a: None, public_seams(), lambda: datetime(2026, 10, 22, tzinfo=UTC),
        import_root=tmp_path, exposure=StageProjection(tmp_path / 'store'),
    )
    labels = LabelStore(store)
    owner = Principal('owner', PrincipalType.USER, frozenset({'owner'}))

    def build():
        return exports.build_dataset(
            store, labels, owner, (),
            exports.ExportInputs(dataset_version='test', eligibility=policy.ELIGIBILITY_LEDGER_V0,
                                 rights=policy.rights_registry_v0(()), now=store._clock()),
            'internal_evaluation', exports.RaterPseudonyms(store.root / 'private'),
            build_snapshot=lambda *a: None, review_payload=lambda *a: {},
        )

    return store, labels, build


@pytest.fixture
def published(tmp_path):
    import shutil

    source = Path(__file__).resolve().parents[2] / 'data/dataset-v0-public'
    return Path(shutil.copytree(source, tmp_path / 'published'))
