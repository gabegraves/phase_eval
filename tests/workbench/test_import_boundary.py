"""Unimportable internals, enforced (design deltas D10, AXA-43zwk.26). An ast walk over the
workbench, the API and the workbench scripts fails when a module

- imports a `_`-prefixed name from a workbench module other than the store primitives the
  ledgers already share (the literal allowlist below: a new private import fails here instead of
  joining the list silently);
- under src/axa, other than seams.py, imports axa_private;
- imports a name from triage or qualification that is not in that module's __all__.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TREES = ("src/phaseeval/workbench", "src/phaseeval/api", "scripts/workbench")
# Store primitives shared by the ledgers and the store scripts at 7c6422f12. Literal on purpose.
SHARED_STORE_PRIVATES = frozenset({"_publish_once", "_read_bytes", "_directory", "_HEX64", "_atomic_write_bytes"})
PUBLIC_SURFACES = ("triage", "qualification")


def _exports(module: str) -> frozenset[str] | None:
    path = REPO / "src/phaseeval/workbench" / f"{module}.py"
    if not path.exists():
        return None
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return frozenset(ast.literal_eval(node.value))
    return frozenset()


def _workbench_module(path: Path, node: ast.ImportFrom) -> str | None:
    """The workbench module an import names, or None when it names something else."""
    if node.level == 1 and path.parent.name == "workbench" and "src" in path.parts:
        return node.module
    if node.level == 0 and node.module and node.module.startswith("phaseeval.workbench."):
        return node.module.removeprefix("phaseeval.workbench.")
    return None


def violations(path: Path, source: str) -> list[str]:
    found = []
    under_src = "src" in path.parts
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            named = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if under_src and path.name != "seams.py" and any(n.split(".")[0] == "axa_private" for n in named):
                found.append(f"{path}: imports axa_private outside seams.py")
        if not isinstance(node, ast.ImportFrom):
            continue
        module = _workbench_module(path, node)
        if module is None:
            continue
        for alias in node.names:
            if alias.name.startswith("_") and not (module == "store" and alias.name in SHARED_STORE_PRIVATES):
                found.append(f"{path}: imports private {module}.{alias.name}")
            exports = _exports(module) if module in PUBLIC_SURFACES else None
            if exports is not None and alias.name not in exports:
                found.append(f"{path}: imports {module}.{alias.name}, which is not in its __all__")
    return found


def _sources() -> list[Path]:
    return sorted(p for tree in TREES for p in (REPO / tree).rglob("*.py"))


def test_no_module_crosses_the_import_boundary() -> None:
    found = [v for path in _sources() for v in violations(path.relative_to(REPO), path.read_text())]
    assert found == []


@pytest.mark.parametrize(
    ("relative", "source", "message"),
    [
        ("src/phaseeval/workbench/exports.py", "from .labels import _append\n", "imports private labels._append"),
        ("src/phaseeval/workbench/store.py", "from .labels import _append\n", "imports private labels._append"),
        ("scripts/workbench/x.py", "from phaseeval.workbench.store import _load\n", "imports private store._load"),
        ("src/phaseeval/workbench/exports.py", "import axa_private\n", "imports axa_private outside seams.py"),
        ("src/phaseeval/api/app.py", "from axa_private.seams import build\n", "imports axa_private outside seams.py"),
        (
            "src/phaseeval/workbench/exports.py",
            "from .qualification import _not_model_judge\n",
            "imports qualification._not_model_judge, which is not in its __all__",
        ),
    ],
)
def test_a_synthetic_crossing_fails(relative: str, source: str, message: str) -> None:
    assert any(message in v for v in violations(Path(relative), source))


def test_seams_may_name_axa_private_and_the_ledgers_may_share_store_primitives() -> None:
    assert violations(Path("src/phaseeval/workbench/seams.py"), "import axa_private\n") == []
    assert violations(Path("src/phaseeval/workbench/labels.py"), "from .store import _publish_once, _read_bytes\n") == []
