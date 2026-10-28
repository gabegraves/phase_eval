"""rl/replay: a compatibility smoke test for reward v0, not a scientific training environment.
Its completion is a phase list with no amounts, so the answer-equals-fit term (K5) can never be
earned (ledger E11; requirements audit 2026-10-27)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl/replay"))

from phaseeval_replay import reward_v0_replay  # noqa: E402

INFO = {"recorded_terms": {"answered": 1.0, "k2_residual": 1.0, "k1_refinement": 1.0, "k5_amounts": 1.0, "reference_phase_list": 0.0}}
ANSWER = json.dumps(["Bi2O3", "Li2CO3", "NiO"])


def test_a_phase_list_with_no_amounts_never_earns_the_amounts_term() -> None:
    completion = [{"role": "assistant", "content": '{"phases": ["NiO", "Li2CO3", "Bi2O3"]}'}]
    # answered 0.20 + K2 0.15 + K1 0.10; K5 (0.15) needs amounts the completion never reports
    assert reward_v0_replay(completion, ANSWER, INFO) == 0.45


def test_a_different_phase_set_inherits_no_saved_fit_term() -> None:
    completion = [{"role": "assistant", "content": '{"phases": ["NiO", "Bi2O3"]}'}]
    assert reward_v0_replay(completion, ANSWER, INFO) == 0.2
