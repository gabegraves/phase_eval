@echo off
rem PhaseEval on Windows: double-click in Explorer. Installs uv if it is missing, then runs the
rem launcher (src\phaseeval\launcher.py) from its locked environment with its pinned Python.
rem Close this window or press Ctrl+C to stop PhaseEval.
setlocal
set "HERE=%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, the Python manager PhaseEval runs on ^(https://docs.astral.sh/uv/^)...
  powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
uv run --project "%HERE%launcher" --locked python "%HERE%launcher\run.py" %*
if errorlevel 1 pause
