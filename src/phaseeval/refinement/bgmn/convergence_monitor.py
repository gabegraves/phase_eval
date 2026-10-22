"""BGMN convergence plateau detection for early termination.

Monitors per-iteration Rwp values from BGMN stdout/protocol output.
Signals termination when DeltaRwp < threshold for N consecutive iterations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# BGMN prints "Rwp=X.XX%" in iteration output when PROTOKOLL=Y
_RWP_PATTERN = re.compile(r"Rwp=(\d+(?:\.\d+)?)%")


@dataclass
class ConvergenceConfig:
    """Configuration for convergence plateau detection.

    Attributes:
        delta_rwp_threshold: Minimum Rwp improvement (percentage points)
            to count as "still converging". Default 0.1pp.
        patience: Number of consecutive sub-threshold iterations before
            declaring plateau. Default 5.
        min_iterations: Never terminate before this many iterations,
            even if plateau is detected. Default 15.
        enabled: Master switch for convergence monitoring.
    """

    delta_rwp_threshold: float = 0.1
    patience: int = 5
    min_iterations: int = 15
    enabled: bool = True


def parse_rwp_from_line(line: str) -> float | None:
    """Extract Rwp value from a BGMN stdout line.

    Returns:
        Rwp as float, or None if line does not contain Rwp.
    """
    m = _RWP_PATTERN.search(line)
    if m:
        return float(m.group(1))
    return None


class ConvergenceMonitor:
    """Track Rwp convergence across BGMN iterations.

    Usage:
        monitor = ConvergenceMonitor(config)
        for each iteration output line:
            rwp = parse_rwp_from_line(line)
            if rwp is not None:
                monitor.update(rwp)
            if monitor.should_terminate():
                kill_process()
    """

    def __init__(self, config: ConvergenceConfig) -> None:
        self._config = config
        self._rwp_history: list[float] = []
        self._stale_count: int = 0

    def update(self, rwp: float) -> None:
        """Record a new Rwp observation."""
        if self._rwp_history:
            delta = self._rwp_history[-1] - rwp
            if delta < self._config.delta_rwp_threshold:
                self._stale_count += 1
            else:
                self._stale_count = 0
        self._rwp_history.append(rwp)

    def should_terminate(self) -> bool:
        """Return True if convergence has plateaued."""
        if not self._config.enabled:
            return False
        if len(self._rwp_history) < self._config.min_iterations:
            return False
        return self._stale_count >= self._config.patience

    @property
    def iteration_count(self) -> int:
        return len(self._rwp_history)

    @property
    def last_rwp(self) -> float | None:
        return self._rwp_history[-1] if self._rwp_history else None

    @property
    def best_rwp(self) -> float | None:
        return min(self._rwp_history) if self._rwp_history else None
