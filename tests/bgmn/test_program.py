import hashlib
import io
import os
import stat
import tomllib
import zipfile
from datetime import date
from pathlib import Path
from typing import get_args

import pytest
import requests

from phaseeval.refinement.bgmn import installer


@pytest.fixture
def download(monkeypatch, tmp_path):
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    calls = []

    def serve(*, sha256=None, size_bytes=None, extra=None, modes=None):
        files = {f"BGMNwin/{name}": b"\x7fELFtest" for name in ("bgmn", "teil", "eflech", "output", "makegeq")}
        files.update(extra or {})
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for name, content in files.items():
                info = zipfile.ZipInfo(name)
                info.external_attr = ((modes or {}).get(name, stat.S_IFREG | 0o755)) << 16
                archive.writestr(info, content)
        body = data.getvalue()
        row = installer.PlatformArchive(
            platform="linux-x86_64", url="https://example.invalid/bgmn.zip",
            sha256=sha256 if sha256 is not None else hashlib.sha256(body).hexdigest(),
            size_bytes=size_bytes if size_bytes is not None else len(body),
            executables=("bgmn", "teil", "eflech", "output", "makegeq"),
            measured_on=date(2026, 10, 26),
        )
        monkeypatch.setattr(installer, "ARCHIVES", {"linux-x86_64": row})

        def get(*args, **kwargs):
            calls.append(args)
            response = requests.Response()
            response.status_code = 200
            response.raw = io.BytesIO(body)  # no Content-Length: enforce the cap on the stream
            return response

        monkeypatch.setattr(requests, "get", get)
        return installer.BGMNInstaller(tmp_path / "bgmn"), row, calls

    return serve


def test_a_wrong_sha256_is_refused_and_nothing_is_extracted(download, tmp_path):
    program, _, _ = download(sha256="0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        program.install()
    assert not list(tmp_path.rglob("BGMNwin"))
    assert not list(tmp_path.rglob("*.zip"))


def test_an_oversized_archive_is_refused(download, tmp_path):
    program, _, _ = download(size_bytes=32)
    with pytest.raises(RuntimeError, match="size"):
        program.install()
    assert not list(tmp_path.rglob("BGMNwin"))
    assert not list(tmp_path.rglob("*.zip"))


@pytest.mark.skipif(os.name == "nt", reason="Unix execute permission bits")
def test_only_the_listed_files_become_executable(download):
    program, row, _ = download(extra={"BGMNwin/unlisted": b"\x7fELFtest", "BGMNwin/readme": b"text"})
    environment = dict(os.environ)
    directory = program.install()
    for path in directory.iterdir():
        assert bool(path.stat().st_mode & 0o111) == (path.name in row.executables), path
    assert os.environ == environment
    assert program.is_installed()


def test_every_pin_is_measured():
    assert set(installer.ARCHIVES) == set(get_args(installer.Platform))
    for key, row in installer.ARCHIVES.items():
        assert row.platform == key
        assert len(bytes.fromhex(row.sha256)) == 32
        assert row.measured_on <= date.today()
        assert 0 < row.size_bytes <= installer.MAX_ARCHIVE_BYTES
        assert {"bgmn", "teil", "eflech", "output"} <= {Path(p).stem.lower() for p in row.executables}


@pytest.mark.parametrize("name", ["../outside", "/outside", "BGMNwin/../../outside", "BGMNwin/..\\outside", "BGMNwin/file:stream"])
def test_unsafe_archive_paths_are_refused(download, tmp_path, name):
    program, _, _ = download(extra={name: b"bad"})
    with pytest.raises(RuntimeError, match="path"):
        program.install()
    assert not list(tmp_path.rglob("BGMNwin"))


def test_archive_links_are_refused(download, tmp_path):
    program, _, _ = download(extra={"BGMNwin/link": b"../../outside"}, modes={"BGMNwin/link": stat.S_IFLNK | 0o777})
    with pytest.raises(RuntimeError, match="path"):
        program.install()
    assert not list(tmp_path.rglob("BGMNwin"))


def test_macos_archive_metadata_is_not_installed(download):
    program, _, _ = download(extra={"__MACOSX/BGMNwin/._bgmn": b"metadata"})
    program.install()
    assert not (program.install_dir / "__MACOSX").exists()


@pytest.mark.parametrize("missing", ["sha256", "measured_on"])
def test_an_unmeasured_pin_downloads_nothing(download, monkeypatch, missing):
    program, row, calls = download()
    monkeypatch.setattr(installer, "ARCHIVES", {row.platform: row.model_copy(update={missing: None})})
    with pytest.raises(installer.BGMNNotInstalledError, match="measured"):
        program.install()
    assert not calls


def test_explicit_consent_installs_the_pinned_program(download):
    program, _, calls = download()
    found = installer.install_bgmn(program.install_dir, installer.Consent.DOWNLOAD, lambda _: pytest.fail("asked"))
    assert found.directory == program.install_dir
    assert found.binary_sha256 == hashlib.sha256(b"\x7fELFtest").hexdigest()
    assert len(calls) == 1


def test_a_failed_reinstall_keeps_the_existing_install(download):
    program, _, _ = download(sha256="0" * 64)
    directory = program.get_bgmn_dir()
    directory.mkdir(parents=True)
    for name in program._executables:
        (directory / name).write_bytes(b"original")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(RuntimeError, match="sha256"):
        program.install(force=True)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_the_unused_refine_command_and_xrdml_package_are_not_shipped():
    import phaseeval
    import pkgutil

    names = {info.name for info in pkgutil.walk_packages(phaseeval.__path__, "phaseeval.")}
    assert "phaseeval.refine_one" not in names
    assert "phaseeval.data" not in names
    repo = Path(__file__).resolve().parents[2]
    manifest = tomllib.loads((repo / "pyproject.toml").read_text())
    assert "phaseeval-refine" not in manifest["project"].get("scripts", {})


def test_readme_bgmn_import_names_the_run_generation_step():
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    section = readme.split("**BGMN or Profex.**", 1)[1].split("**GSAS-II.**", 1)[0]
    assert "harness/examples/dev006/run.sh" in section
