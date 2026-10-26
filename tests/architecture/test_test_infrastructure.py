import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_no_test_module_imports_another_test_module():
    crossings = []
    for path in (ROOT / 'tests').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                crossings.extend(f'{path.relative_to(ROOT)}:{node.lineno}: {name}' for name in names
                                 if any(part.startswith('test_') for part in name.split('.')))
    assert not crossings, '\n'.join(crossings)


@pytest.mark.parametrize('source', [
    'import tests.workbench.test_tree',
    'from tests.workbench.test_tree import World',
    'from tests.workbench import test_tree',
    'from . import test_tree',
    'from . import test_tree as tree',
])
def test_the_import_guard_checks_modules_and_aliases(tmp_path, monkeypatch, source):
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_case.py').write_text(source)
    monkeypatch.setitem(globals(), 'ROOT', tmp_path)
    with pytest.raises(AssertionError, match='test_tree'):
        test_no_test_module_imports_another_test_module()


def test_the_import_guard_allows_support_modules(tmp_path, monkeypatch):
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_case.py').write_text('from tests.support.store import World')
    monkeypatch.setitem(globals(), 'ROOT', tmp_path)
    test_no_test_module_imports_another_test_module()


def test_the_export_fixture_runs_computed_snapshot_checks():
    result = subprocess.run(
        [sys.executable, '-c', '''
import pytest

class CheckExport:
    def pytest_runtest_call(self, item):
        labeled = item.funcargs['labeled']
        dataset = labeled.build()
        snap = next(s for s in dataset.snapshots if s.origin == labeled.tree.answered)
        codes = {c.code for c in snap.checks}
        assert {'K1', 'K2', 'K5'} <= codes, codes

raise SystemExit(pytest.main([
    '-q', '-p', 'no:cacheprovider',
    'tests/workbench/test_export.py::test_the_export_carries_no_custody_field_or_run_identity',
], plugins=[CheckExport()]))
'''],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout, result.stdout


def test_the_pure_snapshot_checks_run_publicly():
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
         'tests/workbench/test_snapshot.py::test_a_residual_peak_at_exactly_the_flag_height_flags',
         'tests/workbench/test_snapshot.py::test_the_scrub_replaces_paths_and_run_names_and_keeps_formulas_ratios_and_numbers'],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '2 passed' in result.stdout, result.stdout


def test_no_skip_names_private_data(tmp_path):
    for reason, expected, collection in [
        ('custody records absent', 1, False), ('private runs absent', 1, True),
        ('bubblewrap not installed', 0, False), ('GSAS-II not installed', 0, True),
    ]:
        directory = tmp_path / 'custody-private-runs'
        directory.mkdir(exist_ok=True)
        path = directory / 'test_skip.py'
        source = 'import pytest\n'
        if collection:
            source += 'pytest.skip(' + repr(reason) + ', allow_module_level=True)\n'
        else:
            source += 'def test_external():\n    pytest.skip(' + repr(reason) + ')\n'
        path.write_text(source)
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-p', 'tests.conftest', str(path)],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT)))},
        )
        assert result.returncode == (5 if collection and expected == 0 else expected), result.stdout + result.stderr
        assert ('private-data skips are forbidden' in result.stdout) == (expected == 1)
