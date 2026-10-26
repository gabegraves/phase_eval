import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_test_module_imports_another_test_module():
    crossings = []
    for path in (ROOT / 'tests').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ''] if isinstance(node, ast.ImportFrom) else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            )
            crossings.extend(f'{path.relative_to(ROOT)}:{node.lineno}: {name}' for name in names
                             if any(part.startswith('test_') for part in name.split('.')))
    assert not crossings, '\n'.join(crossings)


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
        ('bubblewrap not installed', 0, False),
    ]:
        path = tmp_path / 'test_skip.py'
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
        assert result.returncode == expected, result.stdout + result.stderr
        assert ('private-data skips are forbidden' in result.stdout) == (expected == 1)
