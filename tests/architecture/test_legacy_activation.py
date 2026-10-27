import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def test_legacy_schema_inventory_matches_actual_legacy_modules():
    from phaseeval.kernel.records import LEGACY_SCHEMA_IDS

    found = set()
    for path in Path('src/phaseeval').rglob('*.py'):
        if 'kernel' in path.parts:
            continue
        tree = ast.parse(path.read_text())
        schema_fields = [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                         and isinstance(n.target, ast.Name) and n.target.id == 'schema_version']
        for field in schema_fields:
            found.update(n.value for n in ast.walk(field.annotation) if isinstance(n, ast.Constant)
                         and isinstance(n.value, str) and not n.value.startswith('phaseeval.'))
        if schema_fields:
            importlib.import_module('.'.join(path.with_suffix('').relative_to('src').parts))
    assert LEGACY_SCHEMA_IDS == found | {'review-queue.v1'}


@pytest.mark.parametrize('name,status', [('Conflict', 409), ('Busy', 429), ('TooLarge', 413), ('Denied', 403)])
def test_actual_attempt_route_maps_kernel_errors(name, status):
    from tests.support.store import app
    from phaseeval.api.routes.attempts import authenticate_principal, get_workbench_store
    from phaseeval.kernel import errors

    error = getattr(errors, name)(7) if name == 'Busy' else getattr(errors, name)()

    def roots(principal):
        raise error

    app.dependency_overrides[get_workbench_store] = lambda: SimpleNamespace(roots=roots)
    app.dependency_overrides[authenticate_principal] = lambda: object()
    try:
        response = TestClient(app, raise_server_exceptions=False).get('/api/attempts/roots')
        assert response.status_code == status, response.text
        assert response.json() == {'error': type(error).public_message}
        if status == 429:
            assert response.headers['Retry-After'] == '7'
    finally:
        app.dependency_overrides.clear()


def test_actual_import_run_uses_base_serializer_and_preserves_legacy_bytes():
    from phaseeval.kernel.records import Record, canonical_bytes
    from phaseeval.workbench.model import ImportRun

    assert set(ImportRun.__pydantic_decorators__.model_serializers) == set(Record.__pydantic_decorators__.model_serializers)
    harbor = dict(job_name='job', trial_id='trial', task_sha256='a'*64, pattern_sha256='b'*64, cif_sha256={'cif':'c'*64})
    old = {'verb': 'import_run', 'harbor': harbor}
    for data in (old, {**old, 'external_source': 'gsasii_project'}):
        record = ImportRun.model_validate(data)
        assert record.model_dump(mode='json') == data
        assert canonical_bytes(record).endswith(b'\n')


def test_no_module_imports_a_store_private():
    found = []
    for directory in ('src/phaseeval', 'scripts'):
        for path in Path(directory).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module in ('store', 'phaseeval.workbench.store'):
                    found.extend(f'{path}: {name.name}' for name in node.names if name.name.startswith('_'))
    assert found == []
