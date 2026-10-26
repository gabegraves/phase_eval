from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    package: str
    rank: int
    public: tuple[str, ...]
    third_party: frozenset[str]
    legacy: bool = False


_PYD = frozenset({"pydantic"})
_SCI = frozenset({"numpy", "scipy", "pymatgen"})

LAYERS: tuple[Layer, ...] = (
    Layer("kernel", 0, ("records", "ids", "principal", "errors", "lock", "ledger", "stats", "layers"), _PYD),
    Layer(
        "physics", 1,
        ("results", "recipe", "profiles", "lst", "sav", "dia", "xy", "curve", "cif", "residuals", "stability", "chemistry", "composition"),
        _PYD | _SCI,
    ),
    Layer("bgmn", 2, ("program", "refit"), _PYD | frozenset({"requests"})),
    Layer(
        "tree", 2,
        ("case", "actions", "imports", "outcomes", "attempts", "projection", "access", "store", "execution",
         "episodes", "compare", "seams"),
        _PYD,
    ),
    Layer("ingest", 3, ("case_dir", "run_bundle", "agent_run", "external", "bgmn_bundle", "gsasii"), _PYD | frozenset({"numpy"})),
    Layer(
        "review", 3,
        ("policy", "checks", "snapshots", "labels", "stage", "handles", "derive", "views", "reference", "service",
         "composition", "triage"),
        _PYD | frozenset({"numpy"}),
    ),
    Layer(
        "grading", 4,
        ("reduce", "steps", "inputs", "reward", "scorecard", "dashboard", "render", "qualification", "routing_grade", "replay"),
        _PYD,
    ),
    Layer("agentkit", 4, ("run_agent", "refine_tool", "screen_tool", "case_tool"), _PYD),
    Layer("publish", 5, ("leaks", "dataset", "export", "validate", "submit", "migrate", "schemas"), _PYD | frozenset({"jsonschema"})),
    Layer("runtime", 6, ("settings", "wiring", "refit_executor", "cli"), _PYD),
    Layer(
        "web", 7,
        ("app", "errors", "security", "auth", "routes.attempts", "routes.review", "routes.admin", "routes.pages"),
        _PYD | frozenset({"fastapi", "starlette"}),
    ),
    Layer("launcher", 8, ("serve", "session"), _PYD | frozenset({"fastapi", "starlette", "uvicorn"})),
)

_ANY = frozenset({"*"})
LEGACY: tuple[Layer, ...] = tuple(
    Layer(name, 9, (), _ANY, legacy=True)
    for name in ("api", "workbench", "refinement", "orchestrator", "retrieval")
)

ACTIVE_THROUGH = 9
LAYERS = tuple(layer for layer in LAYERS if layer.package != "launcher" or ACTIVE_THROUGH >= 40)
LANDS_IN: dict[str, str] = {
    "kernel": "P31", "physics": "P33", "bgmn": "P33", "tree": "P34",
    "ingest": "P35", "agentkit": "P35", "review": "P36", "grading": "P37",
    "publish": "P38", "runtime": "P39", "web": "P39", "launcher": "P40",
}
MODULE_LANDS_IN = {f"phaseeval.{layer.package}.{name}": int(LANDS_IN[layer.package][1:])
                   for layer in LAYERS for name in layer.public}
MODULE_LANDS_IN.update({
    "phaseeval.kernel.layers": 9, "phaseeval.publish.leaks": 1, "phaseeval.kernel.lock": 10,
    **{f"phaseeval.kernel.{name}": 11 for name in ("records", "ids", "errors", "ledger", "_disk")},
    **{f"phaseeval.physics.{name}": 31 for name in ("results", "recipe", "profiles", "curve", "dia", "composition")},
    **{f"phaseeval.tree.{name}": 32 for name in ("case", "actions", "imports", "outcomes", "attempts", "projection", "episodes", "seams", "compare")},
    "phaseeval.bgmn.program": 32, "phaseeval.web.auth": 32,
    "phaseeval.review.composition": 34,
    "phaseeval.agentkit.refine_tool": 33, "phaseeval.agentkit.screen_tool": 33,
})
MODULE_LANDS_IN.update({
    **{f"phaseeval.bgmn.{name}": 33 for name in ("_aliases", "_str", "_sav", "_runner", "_validate", "_supervisor")},
    "phaseeval.ingest._gpx_reader": 35,
    **{f"phaseeval.review.{name}": 36 for name in ("_derive_trial", "_derive_external", "_derive_live")},
    "phaseeval.__main__": 39, "phaseeval.launcher.__main__": 40,
    "phaseeval.launcher.serve": 40, "phaseeval.launcher.session": 40,
})

ACTIVE_MODULES = frozenset(m for m, pr in MODULE_LANDS_IN.items() if pr <= ACTIVE_THROUGH)

MAX_MODULE_LINES = 700

PRINCIPAL_ATTRIBUTES: frozenset[str] = frozenset({"principal_type", "roles"})
PRINCIPAL_READERS: frozenset[str] = frozenset({"kernel/principal.py", "tree/access.py"})

CLOSURE_FORBIDS: dict[str, frozenset[str]] = {
    "phaseeval.tree.store": frozenset({"fastapi", "scipy", "pymatgen", "phaseeval.bgmn"}),
    "phaseeval.publish.dataset": frozenset({"fastapi", "scipy", "pymatgen", "phaseeval.bgmn"}),
    "phaseeval.grading.reward": frozenset({"fastapi", "scipy", "pymatgen", "numpy", "phaseeval.bgmn"}),
    "phaseeval.review.service": frozenset({"fastapi", "phaseeval.bgmn"}),
    "phaseeval.runtime.wiring": frozenset({"fastapi"}),
}
