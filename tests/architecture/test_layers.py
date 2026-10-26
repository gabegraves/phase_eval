from __future__ import annotations

import ast
import sys
from pathlib import Path

import phaseeval
from phaseeval.kernel.layers import (
    ACTIVE_MODULES,
    LAYERS,
    LEGACY,
    MAX_MODULE_LINES,
    PRINCIPAL_ATTRIBUTES,
    PRINCIPAL_READERS,
)

PKG = Path(phaseeval.__file__).parent
BY_NAME = {layer.package: layer for layer in (*LAYERS, *LEGACY)}
STDLIB = set(sys.stdlib_module_names)
BUDGET = Path(__file__).with_name("legacy_budget.txt")
MIGRATING = bool(LEGACY)


def _own_package(path: Path) -> str | None:
    rel = path.relative_to(PKG).parts
    return rel[0] if len(rel) > 1 else None


def _violations(path: Path) -> list[str]:
    own = _own_package(path)
    if own is None:
        return []
    layer = BY_NAME.get(own)
    found: list[str] = []
    if layer is not None and layer.legacy:
        return []
    tag = path.relative_to(PKG).as_posix()
    if layer is None:
        return [f"{tag}: R6 coverage: package {own!r} has no Layer in kernel/layers.py"]
    source = path.read_text(encoding="utf-8")
    if len(source.splitlines()) > MAX_MODULE_LINES:
        found.append(f"{tag}: R9 size: over {MAX_MODULE_LINES} lines")
    tree = ast.parse(source)
    if path.name == "__init__.py" and tree.body:
        found.append(f"{tag}: R7 empty initializer: package initializers must be empty")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                and node.attr in PRINCIPAL_ATTRIBUTES and tag not in PRINCIPAL_READERS):
            found.append(f"{tag}: R10 principal: reads .{node.attr}")
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if (len(path.relative_to(PKG).parts) - 1) - (node.level - 1) < 1:
                    found.append(f"{tag}: R5 absolute: relative import leaves the package")
                continue
            names = [(node.module or "", [a.name for a in node.names])]
        elif isinstance(node, ast.Import):
            names = [(a.name, []) for a in node.names]
        else:
            continue
        for module, imported in names:
            top = module.split(".")[0]
            if top != "phaseeval":
                if top and top not in STDLIB and top not in layer.third_party and top != "__future__":
                    found.append(f"{tag}: R4 third-party: {module}")
                continue
            parts = module.split(".")[1:]
            if not parts:
                found.append(f"{tag}: R7 init: import of the bare package")
                continue
            target, rest = parts[0], ".".join(parts[1:])
            if target == own:
                continue
            target_layer = BY_NAME.get(target)
            if target_layer is None:
                found.append(f"{tag}: R6 coverage: imports unknown package {target!r}")
                continue
            if target_layer.rank >= layer.rank:
                found.append(f"{tag}: R1 rank: {module} ({target_layer.rank} >= {layer.rank})")
            if any(seg.startswith("_") for seg in parts[1:]) or any(n.startswith("_") for n in imported):
                found.append(f"{tag}: R3 private: {module}")
            if not rest:
                found.append(f"{tag}: R7 init: {module} imported as a package; name the module")
            elif rest not in target_layer.public:
                found.append(f"{tag}: R2 surface: {module} is not a public module of {target}")
    return found


def _all() -> list[str]:
    out: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        out.extend(_violations(path))
    on_disk = {p.name for p in PKG.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))}
    declared = {layer.package for layer in LAYERS}
    if not MIGRATING:
        out.extend(f"R6 coverage: Layer {name!r} has no directory" for name in sorted(declared - on_disk))
    out.extend(f"R6 coverage: directory {name!r} has no Layer" for name in sorted(on_disk - set(BY_NAME)))
    return out


def legacy_files() -> int:
    return sum(1 for layer in LEGACY if (PKG / layer.package).exists() for _ in (PKG / layer.package).rglob("*.py"))


def test_the_direction_rule_holds() -> None:
    assert _all() == []


def test_legacy_only_shrinks() -> None:
    budget = int(BUDGET.read_text()) if BUDGET.exists() else 0
    assert legacy_files() <= budget, "legacy code grew: move the new code into a layer instead"


def test_every_public_module_exists() -> None:
    """A declared active module must exist; each creating PR explicitly activates it."""
    missing = [
        f"{layer.package}.{name}"
        for layer in LAYERS
        for name in layer.public
        if (not MIGRATING or f"phaseeval.{layer.package}.{name}" in ACTIVE_MODULES)
        if not (PKG / layer.package / (name.replace(".", "/") + ".py")).exists()
    ]
    assert missing == []
