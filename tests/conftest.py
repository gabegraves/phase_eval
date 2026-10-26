import re

import pytest

from tests.support.demo import Demo
from tests.support.fake_bgmn import FakeRefine
from tests.support.store import RATER_A, RATER_B, Bench, World

_PRIVATE_SKIP = re.compile(r'custody|private.{0,20}(?:data|run|record)|U(?:4|6|16).{0,20}(?:run|store)|harness v2 trials', re.I)


def pytest_configure(config):
    config.addinivalue_line('markers', 'integration: exercises an external program')
    config.addinivalue_line('markers', 'slow: runs an external refinement')
    config.option.strict_markers = True
    _PRIVATE_SKIPS.clear()


def pytest_runtest_logreport(report):
    if report.skipped and not hasattr(report, 'wasxfail'):
        reason = report.longrepr[2]
        if _PRIVATE_SKIP.search(reason):
            _PRIVATE_SKIPS.append(report.nodeid + ': ' + reason)


_PRIVATE_SKIPS = []
pytest_collectreport = pytest_runtest_logreport


def pytest_sessionfinish(session, exitstatus):
    if _PRIVATE_SKIPS:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        session.config.pluginmanager.get_plugin('terminalreporter').write_sep('=', 'private-data skips are forbidden')
        for skip in _PRIVATE_SKIPS:
            session.config.pluginmanager.get_plugin('terminalreporter').write_line(skip)


@pytest.fixture
def demo_store(tmp_path):
    return Demo(tmp_path / 'demo')


@pytest.fixture
def fake_bgmn():
    return FakeRefine()


@pytest.fixture
def world(tmp_path, fake_bgmn):
    return World(tmp_path / 'store', refine=fake_bgmn)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    from phaseeval.api.app import app

    found = Bench(tmp_path)
    monkeypatch.setattr(app.state, 'workbench_store', found.store, raising=False)
    monkeypatch.setattr(app.state, 'attempt_authenticator', found.authenticator, raising=False)
    return found


@pytest.fixture
def assigned(bench):
    assert bench.assign(['rater-a', 'rater-b']).status_code == 201
    bench.allow(RATER_A)
    bench.allow(RATER_B)
    return bench
