"""Refusal paths of scripts/submit_run.py. No network: every case is refused before any upload."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "data" / "dataset-v0-public" / "episodes.jsonl"
spec = importlib.util.spec_from_file_location("submit_run", REPO / "scripts" / "submit_run.py")
submit_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit_run)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")

    import socket

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)


def seed_record() -> dict:
    return json.loads(SEED.read_text().splitlines()[0])


def write(tmp_path: Path, record) -> Path:
    path = tmp_path / "export.jsonl"
    path.write_text(json.dumps(record) + "\n")
    return path


def run(path: Path, *extra: str) -> int:
    return submit_run.main([str(path), "--name", "phaseeval-test", *extra])


def test_missing_rights_flag_refused(tmp_path, capsys):
    assert run(write(tmp_path, seed_record())) == 2
    assert "--i-hold-the-rights" in capsys.readouterr().err


def test_invalid_file_refused(tmp_path, capsys):
    assert run(write(tmp_path, {"schema_version": "nope"}), "--i-hold-the-rights") == 2
    assert "does not validate" in capsys.readouterr().err


def test_not_json_refused(tmp_path, capsys):
    path = tmp_path / "export.jsonl"
    path.write_text("{not json\n")
    assert run(path, "--i-hold-the-rights") == 2
    assert "not JSON" in capsys.readouterr().err


def test_seed_record_passes_checks(tmp_path):
    assert len(submit_run.check(write(tmp_path, seed_record()), include_scans=False)) == 1


@pytest.mark.parametrize("leak", ["10.0.0.12", "/home/alice/run", "/Users/bob/x.xy", "box.tail1234.ts.net"])
def test_host_leak_refused_naming_field(tmp_path, capsys, leak):
    record = seed_record()
    record["rights"][0]["redistribution"]["observation"] = leak
    assert run(write(tmp_path, record), "--i-hold-the-rights") == 2
    err = capsys.readouterr().err
    assert "line 1.rights[0].redistribution.observation: contains" in err


def test_scan_bytes_refused_without_flag(tmp_path, capsys):
    record = seed_record()
    record["rights"][0]["training_use"]["observation"] = "QUJD" * 2000
    with pytest.raises(submit_run.Refused, match="scan/pattern bytes"):
        submit_run.check(write(tmp_path, record), include_scans=False)
