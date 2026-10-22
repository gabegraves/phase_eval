"""
BGMN subprocess execution with timeout and restart handling.

Runs the BGMN refinement engine as a subprocess with:
- Configurable timeout (soft estimate + hard kill at 2x)
- Automatic restart on crash/timeout (up to max_restarts)
- All failures logged as ERROR unconditionally (never gated by verbose)

Reference: dara-main/src/dara/bgmn_worker.py
"""

from __future__ import annotations

import logging
import os
import platform
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from phaseeval.refinement.bgmn.installer import BGMNInstaller

logger = logging.getLogger(__name__)

_BGMN_RUNTIME_CFG_SUFFIX = ".cfg"


def _failure_excerpt(*, stdout: str, stderr: str, max_chars: int = 500) -> str:
    """Return the most informative crash excerpt, preferring stderr over stdout."""
    stderr = stderr.strip()
    if stderr:
        return stderr[:max_chars]

    stdout = stdout.strip()
    if not stdout:
        return ""

    return stdout[-max_chars:]


class BGMNTimeoutError(Exception):
    """Raised when BGMN refinement exceeds the allowed timeout.

    Attributes:
        elapsed: Seconds elapsed before timeout.
        control_file: Path to the .sav control file.
    """

    def __init__(
        self,
        message: str,
        elapsed: float | None = None,
        control_file: Path | None = None,
    ):
        super().__init__(message)
        self.elapsed = elapsed
        self.control_file = control_file


class BGMNCrashError(Exception):
    """Raised when BGMN process exits with a non-zero return code.

    Attributes:
        return_code: Process exit code.
        stdout: Captured standard output.
        stderr: Captured standard error.
        control_file: Path to the .sav control file.
    """

    def __init__(
        self,
        message: str,
        return_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        control_file: Path | None = None,
    ):
        super().__init__(message)
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.control_file = control_file


class BGMNRunner:
    """Execute BGMN refinement as a subprocess with timeout and restart.

    Uses BGMNInstaller to locate the BGMN executable and set environment
    variables (EFLECH). All failures are logged at ERROR level
    unconditionally per locked decision.

    Args:
        installer: BGMNInstaller instance. If None, creates a default one.
    """

    def __init__(self, installer: BGMNInstaller | None = None) -> None:
        if installer is None:
            from phaseeval.refinement.bgmn.installer import BGMNInstaller

            installer = BGMNInstaller()
        self._installer = installer

    def _prepare_shadow_bgmnwin(self, working_dir: Path, bgmn_path: Path) -> Path:
        """Create a per-job BGMNwin view without shared runtime cfg files."""
        get_bgmn_dir = getattr(self._installer, "get_bgmn_dir", None)
        source_dir = Path(get_bgmn_dir()) if get_bgmn_dir is not None else bgmn_path.parent
        shadow_dir = working_dir / "bgmnwin"
        shadow_dir.mkdir(exist_ok=True)

        if not source_dir.is_dir():
            target_path = shadow_dir / bgmn_path.name
            if not target_path.exists() and not target_path.is_symlink():
                target_path.symlink_to(bgmn_path)
            return shadow_dir

        for source_path in source_dir.iterdir():
            if source_path.name.lower().endswith(_BGMN_RUNTIME_CFG_SUFFIX):
                continue
            target_path = shadow_dir / source_path.name
            if target_path.exists() or target_path.is_symlink():
                continue
            target_path.symlink_to(source_path, target_is_directory=source_path.is_dir())

        return shadow_dir

    @staticmethod
    def _subprocess_env(bgmnwin_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        bgmnwin = str(bgmnwin_dir)
        env["EFLECH"] = bgmnwin
        env["PATH"] = bgmnwin + os.pathsep + env.get("PATH", "")
        return env

    def run(
        self,
        control_file: Path,
        timeout_seconds: float | None = None,
        show_progress: bool = False,
        cpu_affinity: str | None = None,
    ) -> Path:
        """Run BGMN refinement on a single control file.

        Args:
            control_file: Path to the .sav control file.
            timeout_seconds: Maximum wall-clock time in seconds. None for no limit.
            show_progress: If True, stream BGMN output to console.
            cpu_affinity: Optional Linux CPU core range for taskset (e.g. "0-7").

        Returns:
            Path to the .lst result file.

        Raises:
            BGMNNotInstalledError: If BGMN is not installed.
            BGMNTimeoutError: If refinement exceeds timeout.
            BGMNCrashError: If BGMN exits with non-zero return code.
        """
        working_dir = control_file.parent.absolute()
        install_bgmn_path = self._installer.get_bgmn_path()
        shadow_bgmnwin = self._prepare_shadow_bgmnwin(working_dir, install_bgmn_path)
        bgmn_path = shadow_bgmnwin / install_bgmn_path.name
        env = self._subprocess_env(shadow_bgmnwin)
        start_time = time.monotonic()
        cmd = [str(bgmn_path), str(control_file.absolute())]
        if cpu_affinity is not None and platform.system() == "Linux":
            cmd = ["taskset", "-c", cpu_affinity] + cmd

        try:
            cp = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=not show_progress,
                check=False,
                timeout=timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start_time
            msg = f"BGMN timed out after {elapsed:.1f}s on {control_file.name}"
            logger.error(msg)
            raise BGMNTimeoutError(
                msg, elapsed=elapsed, control_file=control_file
            ) from exc

        elapsed = time.monotonic() - start_time

        if cp.returncode != 0:
            stdout_str = cp.stdout.decode(errors="replace") if cp.stdout else ""
            stderr_str = cp.stderr.decode(errors="replace") if cp.stderr else ""
            excerpt = _failure_excerpt(stdout=stdout_str, stderr=stderr_str)
            msg = f"BGMN crashed (exit code {cp.returncode}) on {control_file.name}: {excerpt}"
            logger.error(msg)
            raise BGMNCrashError(
                msg,
                return_code=cp.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                control_file=control_file,
            )

        # Derive .lst path from control file stem
        lst_path = working_dir / f"{control_file.stem}.lst"
        return lst_path

    def run_with_restart(
        self,
        control_file: Path,
        timeout_seconds: float | None = None,
        max_restarts: int = 2,
        cpu_affinity: str | None = None,
    ) -> Path | None:
        """Run BGMN with automatic restart on failure.

        Attempts run() up to max_restarts + 1 times. On BGMNTimeoutError
        or BGMNCrashError, logs ERROR and retries. If all attempts fail,
        logs final ERROR and returns None.

        Each restart uses the same control file since BGMN can resume
        from partial results.

        Args:
            control_file: Path to the .sav control file.
            timeout_seconds: Maximum wall-clock time per attempt.
            max_restarts: Maximum number of retry attempts (default 2).
            cpu_affinity: Optional Linux CPU core range for taskset (e.g. "0-7").

        Returns:
            Path to .lst result file, or None if all attempts failed.
        """
        max_attempts = max_restarts + 1
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self.run(
                    control_file,
                    timeout_seconds=timeout_seconds,
                    cpu_affinity=cpu_affinity,
                )
            except (BGMNTimeoutError, BGMNCrashError) as e:
                last_error = e
                logger.error(
                    "BGMN attempt %d/%d failed for %s: %s",
                    attempt,
                    max_attempts,
                    control_file.name,
                    e,
                )

        logger.error(
            "BGMN refinement failed after %d attempts for %s: %s",
            max_attempts,
            control_file.name,
            last_error,
        )
        return None

    def run_with_monitor(
        self,
        control_file: Path,
        timeout_seconds: float | None = None,
        cpu_affinity: str | None = None,
        on_stdout_line: Callable[[str], bool] | None = None,
    ) -> Path:
        """Run BGMN with line-by-line stdout monitoring via Popen.

        Like run(), but uses Popen instead of subprocess.run() to enable
        real-time stdout monitoring for convergence detection.

        Args:
            control_file: Path to the .sav control file.
            timeout_seconds: Maximum wall-clock time in seconds.
            cpu_affinity: Optional Linux CPU core range for taskset.
            on_stdout_line: Callback invoked with each stdout line.
                If callback returns True, the process is terminated early
                (convergence plateau detected).

        Returns:
            Path to the .lst result file.

        Raises:
            BGMNNotInstalledError: If BGMN is not installed.
            BGMNTimeoutError: If refinement exceeds timeout.
            BGMNCrashError: If BGMN exits with non-zero return code.
        """
        working_dir = control_file.parent.absolute()
        install_bgmn_path = self._installer.get_bgmn_path()
        shadow_bgmnwin = self._prepare_shadow_bgmnwin(working_dir, install_bgmn_path)
        bgmn_path = shadow_bgmnwin / install_bgmn_path.name
        env = self._subprocess_env(shadow_bgmnwin)
        start_time = time.monotonic()
        cmd = [str(bgmn_path), str(control_file.absolute())]
        if cpu_affinity is not None and platform.system() == "Linux":
            cmd = ["taskset", "-c", cpu_affinity] + cmd

        proc = subprocess.Popen(
            cmd,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        terminated_early = False
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        stdout_queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        stderr_queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        terminated_flag = [False]

        def _stream_reader(
            stream: TextIO | None,
            sink: queue.SimpleQueue[str | None],
        ) -> None:
            if stream is None:
                sink.put(None)
                return
            try:
                for line in iter(stream.readline, ""):
                    sink.put(line)
            finally:
                sink.put(None)

        def _drain_queue(
            sink: queue.SimpleQueue[str | None],
            lines: list[str],
            *,
            on_line: Callable[[str], bool] | None = None,
        ) -> bool:
            closed = False
            while True:
                try:
                    item = sink.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    closed = True
                    continue
                lines.append(item)
                if on_line is not None:
                    should_kill = on_line(item)
                    if should_kill and not terminated_flag[0]:
                        logger.info(
                            "Early termination requested for %s after %.1fs",
                            control_file.name,
                            time.monotonic() - start_time,
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        terminated_flag[0] = True
            return closed

        stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(proc.stdout, stdout_queue),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(proc.stderr, stderr_queue),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            stdout_closed = False
            stderr_closed = False
            while True:
                # Check timeout
                elapsed = time.monotonic() - start_time
                if timeout_seconds is not None and elapsed > timeout_seconds:
                    proc.kill()
                    proc.wait()
                    msg = f"BGMN timed out after {elapsed:.1f}s on {control_file.name}"
                    logger.error(msg)
                    raise BGMNTimeoutError(msg, elapsed=elapsed, control_file=control_file)

                stdout_closed = (
                    _drain_queue(stdout_queue, stdout_lines, on_line=on_stdout_line)
                    or stdout_closed
                )
                stderr_closed = _drain_queue(stderr_queue, stderr_lines) or stderr_closed
                terminated_early = terminated_flag[0]

                if proc.poll() is not None and stdout_closed and stderr_closed:
                    break

                time.sleep(0.05)

        except Exception:
            proc.kill()
            proc.wait()
            raise

        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        _drain_queue(stdout_queue, stdout_lines)
        _drain_queue(stderr_queue, stderr_lines)

        elapsed = time.monotonic() - start_time
        stderr_str = "".join(stderr_lines)

        # If we terminated early, treat it as success (BGMN writes .lst on terminate)
        if not terminated_early and proc.returncode != 0:
            stdout_str = "".join(stdout_lines)
            excerpt = _failure_excerpt(stdout=stdout_str, stderr=stderr_str)
            msg = f"BGMN crashed (exit code {proc.returncode}) on {control_file.name}: {excerpt}"
            logger.error(msg)
            raise BGMNCrashError(
                msg,
                return_code=proc.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                control_file=control_file,
            )

        lst_path = working_dir / f"{control_file.stem}.lst"
        return lst_path

    @staticmethod
    def estimate_runtime(n_phases: int, n_points: int) -> float:
        """Estimate refinement runtime based on problem size.

        Calibrated for real PXRD data on 64-core EPYC.
        Soft timeout follows ``max(45, n_phases * n_points * 0.012)``.
        The hard kill threshold is handled by ``compute_timeout()``.

        Args:
            n_phases: Number of phases being refined simultaneously.
            n_points: Number of data points in the diffraction pattern.

        Returns:
            Estimated runtime in seconds.
        """
        return max(45.0, n_phases * n_points * 0.012)

    @staticmethod
    def compute_timeout(
        n_phases: int, n_points: int, *, multiplier: float = 1.5
    ) -> tuple[float, float]:
        """Compute soft and hard timeout values.

        Soft timeout: calibrated runtime estimate.
        Hard timeout: multiplier * soft timeout (hard kill threshold).

        Args:
            n_phases: Number of phases.
            n_points: Number of data points.
            multiplier: Hard timeout = soft * multiplier. Default 1.5 (was 2.0).

        Returns:
            Tuple of (soft_timeout, hard_timeout) in seconds.
        """
        soft = BGMNRunner.estimate_runtime(n_phases, n_points)
        return (soft, multiplier * soft)
