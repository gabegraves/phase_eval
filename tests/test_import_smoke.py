import importlib
import importlib.abc
import pkgutil
import sys
from pathlib import Path

import pytest

PRIVATE_ROOTS = ("detrix", "detrix_core", "axa")


class _BlockPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in PRIVATE_ROOTS:
            raise ImportError(f"private module blocked: {fullname}")
        return None


@pytest.fixture
def private_modules_blocked(monkeypatch):
    for name in list(sys.modules):
        if name.split(".")[0] in (*PRIVATE_ROOTS, "phaseeval"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BlockPrivate(), *sys.meta_path])


def test_every_module_imports_without_private_packages(private_modules_blocked):
    package = importlib.import_module("phaseeval")
    names = [info.name for info in pkgutil.walk_packages(package.__path__, "phaseeval.")]
    for name in names:
        importlib.import_module(name)

    assert "phaseeval.refinement.bgmn.backend" in names
    assert "phaseeval.refine_one" in names
    assert not [name for name in sys.modules if name.split(".")[0] in PRIVATE_ROOTS]


def test_bgmn_is_not_bundled():
    from phaseeval.refinement.bgmn.installer import BGMNInstaller

    package = importlib.import_module("phaseeval")
    shipped = {
        path.name for root in package.__path__ for path in Path(root).rglob("*") if path.is_file()
    }
    assert not shipped & {*BGMNInstaller.EXECUTABLES_UNIX, *BGMNInstaller.EXECUTABLES_WIN}
