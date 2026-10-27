from pathlib import Path

import pytest

from phaseeval.workbench import cli


NAMES = {
    'store': ('STORE', 'WORKBENCH_ROOT'),
    'principals': ('PRINCIPALS', 'API_PRINCIPALS_FILE'),
    'dataset': ('DATASET', 'WORKBENCH_DATASET'),
    'import_root': ('IMPORT_ROOT', 'WORKBENCH_IMPORT_ROOT'),
    'bgmn': ('BGMN_DIR', 'BGMN_DIR'),
    'custody': ('CUSTODY', 'WORKBENCH_CUSTODY'),
}


def test_axa_names_still_work_and_phaseeval_names_win(tmp_path):
    legacy = {f'AXA_{old}': str(tmp_path / field) for field, (_, old) in NAMES.items()}
    with pytest.warns(FutureWarning, match="use PHASEEVAL_"):
        settings = cli.Settings.from_env(legacy)
    for field in NAMES:
        assert getattr(settings, field) == tmp_path / field
    current = {f'PHASEEVAL_{new}': str(tmp_path / f'new-{field}') for field, (new, _) in NAMES.items()}
    settings = cli.Settings.from_env(legacy | current)
    for field in NAMES:
        assert getattr(settings, field) == tmp_path / f'new-{field}'
    with pytest.warns(FutureWarning, match='use PHASEEVAL_'):
        assert cli.Settings.from_env(legacy | {'PHASEEVAL_CUSTODY': ''}).custody is None


@pytest.mark.parametrize('host', ['0.0.0.0', '::', 'localhost', '192.168.1.1'])
def test_bind_host_cannot_be_a_wildcard(host):
    with pytest.raises(ValueError):
        cli.Settings.from_env({'PHASEEVAL_BIND_HOST': host})


@pytest.mark.parametrize('host', ['127.0.0.1', '::1'])
def test_bind_host_accepts_loopback(host):
    assert cli.Settings.from_env({'PHASEEVAL_BIND_HOST': host}).bind_host == host


@pytest.mark.parametrize('field', ['store', 'principals', 'dataset', 'import_root', 'bgmn', 'custody'])
def test_settings_refuse_relative_paths(field, tmp_path):
    values = {'store': tmp_path / 'store', 'principals': tmp_path / 'principals.json', field: Path('relative')}
    with pytest.raises(ValueError):
        cli.Settings(**values)


def test_store_and_principals_must_be_configured_together(tmp_path):
    with pytest.raises(ValueError):
        cli.Settings.from_env({'PHASEEVAL_STORE': str(tmp_path)})


@pytest.mark.parametrize('host', ['0.0.0.0', '::'])
def test_launcher_refuses_a_wildcard_before_opening_a_socket(host):
    from phaseeval.launcher import bind

    with pytest.raises(ValueError, match='loopback'):
        bind(0, host)


def test_launcher_binds_loopback():
    from phaseeval.launcher import bind

    with bind(0, cli.Settings.from_env({}).bind_host) as sock:
        assert sock.getsockname()[0] == '127.0.0.1'


def test_default_launcher_bind_stays_on_loopback():
    from phaseeval.launcher import bind

    with bind(0) as sock:
        assert sock.getsockname()[0] == '127.0.0.1'
