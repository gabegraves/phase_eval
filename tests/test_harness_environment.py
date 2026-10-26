import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name == 'nt', reason='POSIX shell entry point')
def test_dev006_runs_all_four_tools_with_the_root_environment(tmp_path):
    example = tmp_path / 'harness/examples/dev006'
    example.mkdir(parents=True)
    shutil.copyfile(REPO / 'harness/examples/dev006/run.sh', example / 'run.sh')
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    physics = tmp_path / 'harness/physics'
    physics.mkdir()
    for name in ('case_brief', 'candidates', 'screen', 'refine'):
        (physics / f'{name}.py').write_text(f'print({name!r})\n')
    result = subprocess.run(
        ['/bin/sh', str(example / 'run.sh')], cwd=tmp_path,
        env={**os.environ, 'OUT': str(tmp_path / 'output')}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ['case_brief', 'candidates', 'screen', 'refine']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX stale interpreter canary')
def test_agent_tools_use_the_running_interpreter_with_a_stale_launcher_environment(tmp_path):
    script = tmp_path / 'harness/agent/run_agent.py'
    script.parent.mkdir(parents=True)
    shutil.copyfile(REPO / 'harness/agent/run_agent.py', script)
    stale = tmp_path / 'launcher/.venv/bin/python'
    stale.parent.mkdir(parents=True)
    stale.write_text('#!/bin/sh\nexit 97\n')
    stale.chmod(0o755)
    physics = tmp_path / 'harness/physics'
    physics.mkdir()
    (physics / 'case_brief.py').write_text(f'import sys\nprint(sys.executable == {sys.executable!r})\n')
    manifest = tmp_path / 'input/candidates/DEV006/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'candidates': []}))
    runner = runpy.run_path(str(script))['Runner'](SimpleNamespace(
        input_dir=tmp_path / 'input', case='DEV006', out=tmp_path / 'output',
        api_key_env='PHASEEVAL_TEST_API_KEY', base_url='http://unused.invalid',
    ))
    assert runner.call('case_brief', {}) == 'True\n\n[exit 0]'
