@echo off
rem PhaseEval on Windows: double-click in Explorer. Installs uv if it is missing, then runs the
rem launcher (src\phaseeval\launcher.py) from the root locked environment with its pinned Python.
rem Close this window or press Ctrl+C to stop PhaseEval.
setlocal
set "HERE=%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, the Python manager PhaseEval runs on ^(https://docs.astral.sh/uv/^)...
  call powershell -NoProfile -ExecutionPolicy ByPass -Command "$ErrorActionPreference = 'Stop'; $installer = [IO.Path]::GetTempFileName(); try { Invoke-WebRequest -UseBasicParsing -Uri 'https://releases.astral.sh/github/uv/releases/download/0.12.6/uv-installer.ps1' -OutFile $installer; if ((Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash -ne 'b8ac41e4c7744bef23e73f75927586265ee0f38e7cde28cb6c75f11edc3855a3') { throw 'uv installer checksum mismatch; refusing to execute it.' }; $env:UV_NO_MODIFY_PATH = '1'; $env:UV_INSTALL_DIR = Join-Path $env:USERPROFILE '.local\bin'; & ([scriptblock]::Create([IO.File]::ReadAllText($installer))) } finally { Remove-Item -LiteralPath $installer -Force }"
  if errorlevel 1 exit /b 1
)
for /f "tokens=2" %%V in ('uv --version') do set "UV_VERSION=%%V"
for /f "tokens=1-3 delims=." %%A in ("%UV_VERSION%") do (
  if %%A GTR 0 goto run
  if %%A EQU 0 if %%B GTR 12 goto run
  if %%A EQU 0 if %%B EQU 12 if %%C GEQ 6 goto run
)
echo PhaseEval requires uv 0.12.6 or newer. Upgrade uv and try again.
exit /b 1
:run
cd /d "%HERE%"
uv run --locked python -m phaseeval.launcher %*
exit /b %errorlevel%
