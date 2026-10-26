import ast
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def test_one_manifest_declares_every_runtime_import():
    manifest = tomllib.loads((REPO / 'pyproject.toml').read_text())
    declared = {requirement.split('=')[0].split('>')[0].split('<')[0].lower().replace('_', '-')
                for requirement in manifest['project']['dependencies']}
    distributions = {'yaml': 'pyyaml', 'dara': 'dara-xrd', 'multipart': 'python-multipart'}
    imports = set()
    for path in (REPO / 'src/phaseeval').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and not node.level else
                     [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            imports.update(name.split('.')[0] for name in names if name)
    runtime = imports - sys.stdlib_module_names - {'phaseeval', '__future__', 'GSASII', 'GSASIIscriptable'}
    assert {distributions.get(name, name).replace('_', '-') for name in runtime} <= declared
    assert {'dara-xrd', 'jsonschema', 'websockets'} <= declared
    for name in runtime:
        importlib.import_module(name)
    assert importlib.metadata.version('dara-xrd') == '1.1.5'
    assert not (REPO / 'launcher/pyproject.toml').exists()
    assert not (REPO / 'launcher/uv.lock').exists()
    assert not (REPO / 'launcher/run.py').exists()
    assert (REPO / 'uv.lock').is_file()


def test_ci_runs_the_public_suite_and_launcher_on_supported_hosts():
    workflow = yaml.safe_load((REPO / '.github/workflows/ci.yml').read_text())
    jobs = workflow['jobs']
    assert set(jobs['test']['strategy']['matrix']['os']) == {'ubuntu-24.04', 'macos-14'}
    assert set(jobs['launcher']['strategy']['matrix']['os']) == {'ubuntu-24.04', 'macos-14', 'windows-2022'}
    test_steps = json.dumps(jobs['test']['steps'])
    assert 'uv sync --locked' in test_steps
    assert '--import-mode=importlib' in test_steps
    assert 'bubblewrap' in test_steps and 'node' in test_steps
    assert 'tampered' in json.dumps(jobs['launcher']['steps'])
    assert './phaseeval.sh --check' in json.dumps(jobs['fresh-clone']['steps'])
    assert 'gsasii' in jobs


def _script_environment(tmp_path):
    tools = tmp_path / 'bin'
    tools.mkdir()
    env = {**os.environ, 'HOME': str(tmp_path), 'USERPROFILE': str(tmp_path)}
    if os.name == 'nt':
        env['PATH'] = str(tools) + os.pathsep + str(Path(os.environ['SystemRoot']) / 'System32')
    else:
        for name in ('dirname', 'mktemp', 'rm', 'sh', 'chmod', 'sha256sum', 'shasum', 'awk', 'sort', 'head'):
            if location := shutil.which(name):
                (tools / name).symlink_to(location)
        env['PATH'] = str(tools)
    return tools, env


def _script_command():
    return ([os.environ['COMSPEC'], '/d', '/c', str(REPO / 'PhaseEval.bat')] if os.name == 'nt'
            else ['/bin/sh', str(REPO / 'phaseeval.sh')])


def test_the_launch_script_refuses_a_tampered_uv_installer(tmp_path):
    tools, env = _script_environment(tmp_path)
    marker = tmp_path / 'executed'
    if os.name == 'nt':
        powershell = shutil.which('powershell')
        shim = tools / 'download.py'
        shim.write_text(
            'import subprocess, sys\n'
            f'prefix = "function Invoke-WebRequest {{ param($Uri, $OutFile, [switch]$UseBasicParsing); '
            f'Set-Content -Path $OutFile -Value \'Set-Content -Path {marker.as_posix()} -Value executed\' }}; "\n'
            f'sys.exit(subprocess.call([{powershell!r}, "-NoProfile", "-Command", prefix + sys.argv[-1]]))\n'
        )
        (tools / 'powershell.cmd').write_text(f'@"{sys.executable}" "{shim}" %*\n')
    else:
        download = tools / 'curl'
        download.write_text(
            '#!/bin/sh\nwhile [ "$#" -gt 0 ] && [ "$1" != "-o" ]; do shift; done\n'
            '[ "$#" -gt 0 ] || exit 1\nshift\n'
            f'printf "#!/bin/sh\\ntouch {marker}\\n" > "$1"\n'
        )
        download.chmod(0o755)
    result = subprocess.run(_script_command(), env=env, input='\n', capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'checksum' in (result.stdout + result.stderr).lower()
    assert not marker.exists()


def test_the_launch_script_refuses_an_old_uv(tmp_path):
    tools, env = _script_environment(tmp_path)
    if os.name == 'nt':
        (tools / 'uv.cmd').write_text('@echo uv 0.1.0\n')
    else:
        uv = tools / 'uv'
        uv.write_text('#!/bin/sh\necho uv 0.1.0\n')
        uv.chmod(0o755)
    result = subprocess.run(_script_command(), env=env, input='\n', capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'uv 0.12.6 or newer' in result.stdout + result.stderr


@pytest.mark.parametrize('entry', ['native', 'macos'] if os.name != 'nt' else ['native'])
def test_the_launch_script_runs_the_root_manifest_from_any_directory(tmp_path, entry):
    tools, env = _script_environment(tmp_path)
    log = tmp_path / 'invocation.txt'
    env['P09_UV_LOG'] = str(log)
    if os.name == 'nt':
        (tools / 'uv.cmd').write_text(
            '@echo off\nif "%1"=="--version" (echo uv 0.12.6 & exit /b 0)\n'
            'cd > "%P09_UV_LOG%"\necho %*>> "%P09_UV_LOG%"\n'
        )
    else:
        uv = tools / 'uv'
        uv.write_text('#!/bin/sh\nif [ "$1" = --version ]; then echo uv 0.12.6; else\n'
                      'printf "%s\\n" "$PWD" "$@" > "$P09_UV_LOG"\nfi\n')
        uv.chmod(0o755)
    command = ['/bin/sh', str(REPO / 'PhaseEval.command')] if entry == 'macos' else _script_command()
    result = subprocess.run(command + ['--no-browser'], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = log.read_text().splitlines()
    assert Path(lines[0]) == REPO
    assert ' '.join(lines[1:]) == 'run --locked python -m phaseeval.launcher --no-browser'


def test_the_api_module_has_no_development_server_entry_point():
    result = subprocess.run(
        [sys.executable, '-c', 'import runpy, uvicorn; '
         'uvicorn.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("development server started")); '
         'runpy.run_module("phaseeval.api.app", run_name="__main__")'],
        cwd=REPO, env={**os.environ, 'PYTHONPATH': str(REPO / 'src')}, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
