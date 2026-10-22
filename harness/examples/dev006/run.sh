#!/bin/sh
# Run the four harness tools on DEV006 with the COD precursor structures.
# Needs launcher/.venv and BGMN (see the top-level README).
set -e
cd "$(dirname "$0")/../../physics"
IN=../examples/dev006/input
OUT=${OUT:-../examples/dev006/output}
mkdir -p "$OUT"
PYTHONPATH=../../src ../../launcher/.venv/bin/python case_brief.py --input-dir "$IN" --case DEV006
PYTHONPATH=../../src ../../launcher/.venv/bin/python candidates.py --input-dir "$IN" --case DEV006
PYTHONPATH=../../src ../../launcher/.venv/bin/python screen.py --input-dir "$IN" --output-dir "$OUT" --case DEV006 --no-search
PYTHONPATH=../../src ../../launcher/.venv/bin/python refine.py --input-dir "$IN" --output-dir "$OUT" --case DEV006 \
  --candidates cod_1010093 cod_9008283 cod_1526458 --run-label cod-precursors
