#!/bin/sh
# PhaseEval on Linux (and on macOS through PhaseEval.command): install uv if it is missing, then
# run the launcher from the root locked environment with its pinned Python. The launcher itself
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
  installer=$(mktemp)
  trap 'rm -f "$installer"' EXIT HUP INT TERM
  url=https://releases.astral.sh/github/uv/releases/download/0.12.6/uv-installer.sh
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$url" -o "$installer"
  else
    wget -qO "$installer" "$url"
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    checksum=$(sha256sum "$installer")
  else
    checksum=$(shasum -a 256 "$installer")
  fi
  if [ "${checksum%% *}" != a8be80c1970926fc4e642486e7c92f5a3d724ebe121ceff8f638c3e7ca277074 ]; then
    echo "uv installer checksum mismatch; refusing to execute it." >&2
    exit 1
  fi
  UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$HOME/.local/bin" sh "$installer"
  rm -f "$installer"
  trap - EXIT HUP INT TERM
fi
if ! version=$(uv --version) || ! printf '%s\n' "$version" | awk '{split($2, v, "."); ok = ($1 == "uv" && $2 ~ /^[0-9]+\.[0-9]+\.[0-9]+$/ && (v[1] > 0 || (v[1] == 0 && (v[2] > 12 || (v[2] == 12 && v[3] >= 6)))))} END {exit !ok}'; then
  echo "PhaseEval requires uv 0.12.6 or newer. Upgrade uv and try again." >&2
  exit 1
fi
cd "$here"
exec uv run --locked python -m phaseeval.launcher "$@"
