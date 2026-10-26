from __future__ import annotations

import json
import subprocess
import sys

import pytest

from phaseeval.kernel.layers import ACTIVE_MODULES, CLOSURE_FORBIDS, LANDS_IN, LEGACY


def _skip_until_landed(module: str) -> None:
    if LEGACY and module not in ACTIVE_MODULES:
        pytest.skip(f"{module} lands in {LANDS_IN.get(module.split('.')[1], 'a later PR')}")


@pytest.mark.parametrize("module", sorted(CLOSURE_FORBIDS))
def test_import_closure_stays_light(module: str) -> None:
    _skip_until_landed(module)
    code = (
        "import json, sys\n"
        f"sys.path[:0] = {json.dumps([p for p in sys.path if p])}\n"
        f"import {module}\n"
        "print(json.dumps(sorted(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True)
    loaded = set(json.loads(out.stdout))
    tops = {name.split(".")[0] for name in loaded}
    hit = {f for f in CLOSURE_FORBIDS[module] if f in loaded or f in tops or any(m.startswith(f + ".") for m in loaded)}
    assert hit == set(), f"import {module} loaded {sorted(hit)}"
