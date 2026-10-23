"""Smoke test: load_environment, then score each public episode's recorded answer and two
counterfactual completions with the environment's own rubric (no model call)."""

import asyncio
import json
import sys

import verifiers as vf

dataset_dir = sys.argv[1]
env = vf.load_environment("phaseeval-replay", dataset_dir=dataset_dir)
print("env:", type(env).__name__, "rows:", len(env.dataset))

from phaseeval_replay import reward_v0_replay as fn
for row in env.dataset:
    info, answer = row["info"], row["answer"]
    recorded = [{"role": "assistant", "content": json.dumps({"phases": json.loads(answer)})}]
    wrong = [{"role": "assistant", "content": '{"phases": ["NaCl"]}'}]
    empty = [{"role": "assistant", "content": "I am not sure."}]
    scores = [fn(completion=c, answer=answer, info=info) for c in (recorded, wrong, empty)]
    print(info["case_id"], info["episode"][:12], "phases", answer,
          "| scorecard total", info["recorded_total"], "| replay recorded/wrong/none", scores)

leaks = [row["info"]["case_id"] for row in env.dataset
         if any(p in row["prompt"][0]["content"] for p in json.loads(row["answer"]))]
print("prompts that contain a recorded phase formula:", leaks or "none")
