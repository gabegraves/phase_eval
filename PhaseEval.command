#!/bin/sh
# PhaseEval on macOS: double-click in Finder. Terminal opens and runs phaseeval.sh; close that
# window or press Ctrl+C to stop PhaseEval.
exec "$(cd "$(dirname "$0")" && pwd)/phaseeval.sh" "$@"
