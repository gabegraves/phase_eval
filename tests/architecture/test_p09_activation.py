import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', ['kernel.records', 'kernel._disk', '__main__', 'launcher.__main__'])
def test_active_modules_are_mandatory(tmp_path, module):
    package = tmp_path / 'phaseeval'
    for name in ('', 'kernel', 'launcher'):
        directory = package / name
        directory.mkdir(exist_ok=True)
        (directory / '__init__.py').write_text('')
    modules = ('kernel.records', 'kernel._disk', '__main__', 'launcher.__main__')
    (package / 'kernel/layers.py').write_text(
        'from types import SimpleNamespace\n'
        "LAYERS = (SimpleNamespace(package='kernel', public=('records',)),)\n"
        "LEGACY = (SimpleNamespace(package='workbench'),)\n"
        f'ACTIVE_MODULES = frozenset({tuple(f"phaseeval.{name}" for name in modules)!r})\n'
        'MAX_MODULE_LINES = 700\nPRINCIPAL_ATTRIBUTES = PRINCIPAL_READERS = frozenset()\n'
    )
    for name in modules:
        (package / (name.replace('.', '/') + '.py')).write_text('')
    test = tmp_path / 'test_layers.py'
    shutil.copyfile(Path(__file__).with_name(test.name), test)
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               f'{test}::test_every_public_module_exists']
    env = {**os.environ, 'PYTHONPATH': str(tmp_path), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    (package / (module.replace('.', '/') + '.py')).unlink()
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert module in result.stdout, result.stdout
    assert 'AssertionError' in result.stdout, result.stdout


@pytest.mark.parametrize('violation, rule', [
    ('', None),
    ('from phaseeval.workbench.model import Record\n', 'R1 rank'),
    ('from phaseeval.physics.results import StageFacts\n', 'R1 rank'),
    ('from phaseeval.kernel.unknown import Thing\n', 'R2 surface'),
    ('from phaseeval.kernel.records import _private\n', 'R3 private'),
    ('import requests\n', 'R4 third-party'),
    ('from ..kernel.layers import Layer\n', 'R5 absolute'),
    ('import phaseeval.unknown.module\n', 'R6 coverage'),
    ('from phaseeval.kernel import records\n', 'R7 init'),
    ('SIDE_EFFECT = 42\n', 'R7 empty initializer'),
    ('"""Package documentation."""\n', 'R7 empty initializer'),
    ('pass\n' * 700 + 'pass', 'R9 size'),
    ('principal.roles\n', 'R10 principal'),
])
def test_p09_architecture_suite_collects_without_future_modules(tmp_path, violation, rule):
    source = Path(__file__).parents[2]
    package = tmp_path / "phaseeval"
    for name in ("", "kernel", "publish"):
        directory = package / name
        directory.mkdir(exist_ok=True)
        (directory / "__init__.py").write_text("")
    layers = (source / "src/phaseeval/kernel/layers.py").read_text()
    layers = re.sub(r"^ACTIVE_THROUGH = \d+$", "ACTIVE_THROUGH = 9", layers, flags=re.M)
    (package / "kernel/layers.py").write_text(layers)
    shutil.copyfile(source / "src/phaseeval/publish/leaks.py", package / "publish/leaks.py")
    tests = tmp_path / "checks"
    tests.mkdir()
    for name in ("layers", "audience", "records", "tables", "closure"):
        shutil.copyfile(Path(__file__).with_name(f"test_{name}.py"), tests / f"test_{name}.py")
    (tests / "legacy_budget.txt").write_text("0\n")
    if violation:
        path = package / ('kernel' if rule.startswith(('R1', 'R4', 'R5')) else 'publish') / 'violation.py'
        if rule == 'R7 empty initializer':
            path = package / 'publish/__init__.py'
        path.write_text(violation)
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tests)]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == (1 if violation else 0), result.stdout + result.stderr
    if violation:
        assert rule in result.stdout, result.stdout
        return
    assert "passed" in result.stdout
    assert "skipped" in result.stdout
    (package / "publish/leaks.py").unlink()
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "publish.leaks" in result.stdout


@pytest.mark.parametrize('order, expected', [("('K1',)", 0), ("('K2',)", 1)])
def test_field_order_check_activates_at_p20(tmp_path, order, expected):
    package = tmp_path / 'phaseeval'
    for name in ('', 'kernel', 'workbench'):
        directory = package / name
        directory.mkdir(exist_ok=True)
        (directory / '__init__.py').write_text('')
    (package / 'kernel/layers.py').write_text(
        'ACTIVE_MODULES = ()\nACTIVE_THROUGH = 20\nLEGACY = True\nMODULE_LANDS_IN = {}\n'
    )
    (package / 'workbench/snapshot.py').write_text(
        "class CheckSet:\n    model_fields = {'k1': None, 'policy': None}\n"
        f'CHECK_ORDER = {order}\n'
    )
    (package / 'workbench/triage.py').write_text("raise AssertionError('policy imported before P21')\n")
    test = tmp_path / 'test_tables.py'
    shutil.copyfile(Path(__file__).with_name(test.name), test)
    result = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
         f'{test}::test_check_order_is_the_check_set_fields'],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(tmp_path)}, capture_output=True, text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert ('1 failed' if expected else '1 passed') in result.stdout, result.stdout
