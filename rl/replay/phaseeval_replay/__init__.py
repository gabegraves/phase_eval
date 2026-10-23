"""PhaseEval replay environment for verifiers (prototype, unpublished).

Inputs are public only: the phase_eval repository (REWARD_SPEC_V0, the scorecard that applies
it, and data/dataset-v0-public with six Qwen episodes on DEV006 and DEV102).

This is a replay scorer, not an RL environment: the policy has no tools and cannot refine.
One task per public episode. The policy sees only the case id and the declared elements and
must answer with a JSON phase list. The prompt never shows the recorded answer or the
recorded run's tool trace. The deterministic checks
(K1, K2, K5) were measured on the recorded run's saved fit, so they are credited only when the
completion's phase set equals the recorded answer's phase set. A different answer has no fit,
so those terms score 0 (REWARD_SPEC_V0 rule: not computable scores 0). The reference and model
judge terms have no public reading and score 0, as the public scorecard scores them.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset
from pymatgen.core import Composition

from phaseeval.workbench.scorecard import REWARD_SPEC_V0, scorecard_view


def _reduced(formula: str) -> str:
    return Composition(formula.replace(" ", "")).reduced_formula


def _phase_set(formulas: list[str]) -> list[str]:
    return sorted({_reduced(f) for f in formulas})


def episode_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    view = scorecard_view(dataset_dir, reference=lambda _case: None)
    snaps = {}
    for line in (dataset_dir / "snapshots.jsonl").read_text().splitlines():
        s = json.loads(line)
        snaps[s["origin"]] = s
    rows = []
    for e in view.episodes:
        snap = snaps[e.leaf]
        phases = _phase_set([p["formula"] for p in snap["chemistry"]["phases"]])
        elements = snap["chemistry"]["element_balance"]["declared_elements"]
        prompt = (
            f"PXRD case {e.case_id}. Declared elements: {', '.join(elements)}.\n"
            "Name the crystalline phases present in this scan.\n"
            'Answer with JSON only: {"phases": ["<formula>", ...]}'
        )
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "answer": json.dumps(phases),
                "info": {
                    "episode": e.episode,
                    "case_id": e.case_id,
                    "model": e.model,
                    "recorded_terms": {t.id: t.value for t in e.reward.terms},
                    "recorded_total": e.reward.total,
                },
            }
        )
    return rows


def parse_phases(text: str) -> list[str] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        phases = json.loads(match.group(0)).get("phases")
        if not isinstance(phases, list) or not phases:
            return None
        return _phase_set([str(p) for p in phases])
    except (ValueError, AttributeError):
        return None


def reward_v0_replay(completion: Any, answer: str, info: dict[str, Any], **_: Any) -> float:
    text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
    phases = parse_phases(text)
    if phases is None:
        return 0.0
    same = phases == json.loads(answer)
    total = 0.0
    for c in REWARD_SPEC_V0.components:
        if c.id == "answered":
            value = 1.0
        elif c.source == "deterministic_check":
            value = info["recorded_terms"][c.id] if same else 0.0
        else:
            value = 0.0
        total += c.weight * value
    return round(total, 4)


def load_environment(dataset_dir: str | None = None, **kwargs: Any) -> vf.Environment:
    root = Path(dataset_dir or os.environ["PHASEEVAL_DATASET_DIR"])
    rows = episode_rows(root)
    rubric = vf.Rubric(funcs=[reward_v0_replay], weights=[1.0])
    return vf.SingleTurnEnv(dataset=Dataset.from_list(rows), rubric=rubric, **kwargs)
