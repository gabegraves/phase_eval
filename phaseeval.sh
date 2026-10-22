#!/bin/sh
# PhaseEval on Linux (and on macOS through PhaseEval.command): install uv if it is missing, then
# run the launcher from its locked environment with its pinned Python. The launcher itself
# (src/phaseeval/launcher.py) reads launcher/phaseeval.toml and is the same on every OS.
#
#   ./phaseeval.sh                    start, or open the running PhaseEval
#   ./phaseeval.sh --print-link       print a one-time sign-in link instead of opening a browser
#   ./phaseeval.sh --install-desktop  add PhaseEval to the Linux app menu and desktop
set -eu
here=$(cd "$(dirname "$0")" && pwd)

if [ "${1:-}" = "--install-desktop" ]; then
  apps=${XDG_DATA_HOME:-$HOME/.local/share}/applications
  mkdir -p "$apps"
  cat > "$apps/phaseeval.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=PhaseEval
Comment=Review XRD agent analyses on this computer
Exec="$here/phaseeval.sh"
Path=$here
Terminal=true
Categories=Science;
DESKTOP
  chmod +x "$apps/phaseeval.desktop"
  echo "PhaseEval added to $apps"
  desktop=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
  if [ -d "$desktop" ]; then
    cp "$apps/phaseeval.desktop" "$desktop/" && chmod +x "$desktop/phaseeval.desktop"
    gio set "$desktop/phaseeval.desktop" metadata::trusted true 2>/dev/null || true
    echo "PhaseEval added to $desktop"
  fi
  exit 0
fi

PATH="$HOME/.local/bin:$PATH"
export PATH
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv, the Python manager PhaseEval runs on (https://docs.astral.sh/uv/)..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    wget -qO- https://astral.sh/uv/install.sh | sh
  fi
fi
exec uv run --project "$here/launcher" --locked python "$here/launcher/run.py" "$@"
