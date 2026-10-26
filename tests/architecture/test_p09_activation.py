import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


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
    (package / "kernel/layers.py").write_text(layers)
    shutil.copyfile(source / "src/phaseeval/publish/leaks.py", package / "publish/leaks.py")
    tests = tmp_path / "checks"
    tests.mkdir()
    for name in ("layers", "audience", "records", "tables", "closure"):
        shutil.copyfile(Path(__file__).with_name(f"test_{name}.py"), tests / f"test_{name}.py")
    (tests / "legacy_budget.txt").write_text("0\n")
    if violation:
        path = package / ('kernel' if rule.startswith(('R1', 'R4', 'R5')) else 'publish') / 'violation.py'
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
