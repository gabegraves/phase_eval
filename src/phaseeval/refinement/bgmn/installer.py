"""
Managed BGMN installation, validation, and path management.

Downloads BGMN from Berkeley OCF on first use to ~/.axa/bgmn/,
validates executables, and configures environment variables.

Key design decisions:
- Install to ~/.axa/bgmn/ (NOT package-internal like DARA)
- Hard fail with BGMNNotInstalledError (never silent fallback)
- Platform support: Linux + Windows
- Progress bar via tqdm during download

Reference: dara-main/src/dara/bgmn/download_bgmn.py
"""

from __future__ import annotations

import logging
import os
import platform
import stat
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


# ELF, then Mach-O 64-bit, 32-bit and universal headers.
BINARY_MAGIC = (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe")


class BGMNNotInstalledError(Exception):
    """Raised when BGMN is required but not installed.

    Provides actionable installation instructions. Per locked decision:
    no silent fallback when BGMN is missing.
    """

    def __init__(self, message: str | None = None):
        if message is None:
            message = (
                "BGMN is not installed. "
                "Run `axa refinement install-bgmn` or call BGMNInstaller().install()"
            )
        super().__init__(message)


class BGMNInstaller:
    """Managed BGMN download, validation, and path management.

    Downloads BGMN executable from Berkeley OCF to a user-local directory
    (default: ~/.axa/bgmn/) on first use. Supports Linux and Windows.

    Usage:
        installer = BGMNInstaller()
        if not installer.is_installed():
            installer.install()
        bgmn_path = installer.get_bgmn_path()

    Or use the convenience function:
        path = ensure_bgmn()  # Install if needed, return path

    Attributes:
        INSTALL_DIR: Default installation directory (~/.axa/bgmn/).
        BGMN_URL_TEMPLATE: URL template for BGMN zip downloads.
        EXECUTABLES_UNIX: Required executables on Linux/macOS.
        EXECUTABLES_WIN: Required executables on Windows.
    """

    INSTALL_DIR: Path = Path.home() / ".axa" / "bgmn"
    # DARA's current copies (CederGroupHub/dara download_bgmn.py); Darwin_m1 is the arm64 build.
    BGMN_URL_TEMPLATE: str = "https://cedergrouphub.github.io/dara/_static/bgmnwin_{os_name}.zip"
    EXECUTABLES_UNIX: list[str] = ["bgmn", "teil", "eflech", "output"]
    EXECUTABLES_WIN: list[str] = ["bgmn.exe", "teil.exe", "eflech.exe", "output.exe"]

    def __init__(self, install_dir: Path | None = None):
        """Initialize installer.

        Args:
            install_dir: Override default install directory.
                Defaults to $AXA_BGMN_DIR, else ~/.axa/bgmn/.
        """
        configured = os.environ.get("AXA_BGMN_DIR")
        self.install_dir = install_dir or (Path(configured) if configured else self.INSTALL_DIR)

    @property
    def _executables(self) -> list[str]:
        """Return platform-appropriate executable names."""
        if platform.system() == "Windows":
            return self.EXECUTABLES_WIN
        return self.EXECUTABLES_UNIX

    @property
    def _bgmn_win_dir(self) -> Path:
        """Return the BGMNwin subdirectory path."""
        return self.install_dir / "BGMNwin"

    def is_installed(self) -> bool:
        """Check if BGMN is installed with all required executables.

        Returns:
            True if BGMNwin directory exists and contains all required
            executables for the current platform.
        """
        bgmn_dir = self._bgmn_win_dir
        if not bgmn_dir.is_dir():
            return False

        for exe in self._executables:
            if not (bgmn_dir / exe).exists():
                return False
        return True

    def get_bgmn_path(self) -> Path:
        """Return path to the bgmn executable.

        Returns:
            Path to the bgmn (or bgmn.exe on Windows) executable.

        Raises:
            BGMNNotInstalledError: If BGMN is not installed.
        """
        if not self.is_installed():
            raise BGMNNotInstalledError()

        exe_name = "bgmn.exe" if platform.system() == "Windows" else "bgmn"
        return self._bgmn_win_dir / exe_name

    def get_bgmn_dir(self) -> Path:
        """Return the BGMNwin directory path.

        Returns:
            Path to the BGMNwin directory (may not exist yet).
        """
        return self._bgmn_win_dir

    def install(self, force: bool = False) -> Path:
        """Download and install BGMN from DARA's GitHub Pages.

        Downloads the platform-appropriate zip file, extracts it to
        install_dir/BGMNwin/, and sets executable permissions on Unix.

        Args:
            force: If True, re-download even if already installed.

        Returns:
            Path to the BGMNwin directory.

        Raises:
            RuntimeError: If download fails or platform is unsupported.
        """
        if self.is_installed() and not force:
            logger.info("BGMN already installed at %s", self._bgmn_win_dir)
            return self._bgmn_win_dir

        import requests
        from tqdm import tqdm

        os_name = platform.system()
        if os_name not in ("Darwin", "Linux", "Windows"):
            raise RuntimeError(f"Unsupported platform: {os_name}")

        url = self.download_url()
        logger.info("Downloading BGMN from %s", url)

        # Create install directory
        self.install_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.install_dir / "bgmnwin.zip"

        # Download with progress bar
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"Cannot download BGMN from {url} (HTTP {response.status_code}). "
                "Check your internet connection."
            )

        total_size = int(response.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="iB", unit_scale=True, desc="BGMN") as pbar:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    pbar.update(len(chunk))
                    f.write(chunk)

        # Extract
        logger.info("Extracting BGMN to %s", self.install_dir)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(self.install_dir))

        # Remove zip
        zip_path.unlink()

        # Set executable permissions on Unix: every ELF or Mach-O file in the package, since BGMN
        # runs helpers (makegeq among them) beyond the four executables is_installed requires.
        if os_name in ("Linux", "Darwin"):
            for exe_path in self._bgmn_win_dir.iterdir():
                if exe_path.is_file() and exe_path.read_bytes()[:4] in BINARY_MAGIC:
                    exe_path.chmod(
                        exe_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
                    )

        logger.info("BGMN installed successfully at %s", self._bgmn_win_dir)
        return self._bgmn_win_dir

    def download_url(self) -> str:
        """The package for this platform: Darwin_m1 on an arm64 Mac, else the OS name."""
        os_name = platform.system()
        if os_name == "Darwin" and platform.machine() == "arm64":
            os_name = "Darwin_m1"
        return self.BGMN_URL_TEMPLATE.format(os_name=os_name)

    def validate_installation(self) -> dict[str, bool]:
        """Check each required executable and return status dict.

        Returns:
            Dict mapping executable name to whether it exists.
            Example: {"bgmn": True, "teil": True, "eflech": False, "output": True}
        """
        result: dict[str, bool] = {}
        for exe in self._executables:
            result[exe] = (self._bgmn_win_dir / exe).exists()
        return result

    def set_environment(self) -> None:
        """Set EFLECH environment variable and add BGMNwin to PATH.

        BGMN requires EFLECH to point to the BGMNwin directory for
        locating scattering factor files and instrument profiles.

        Raises:
            BGMNNotInstalledError: If BGMN is not installed.
        """
        if not self.is_installed():
            raise BGMNNotInstalledError()

        bgmn_dir = str(self._bgmn_win_dir)
        os.environ["EFLECH"] = bgmn_dir
        if bgmn_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bgmn_dir + os.pathsep + os.environ.get("PATH", "")

        logger.debug("Set EFLECH=%s and added to PATH", bgmn_dir)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def ensure_bgmn(install_dir: Path | None = None) -> Path:
    """Install BGMN if needed and return the BGMNwin directory path.

    Convenience function that creates an installer, checks if BGMN is
    installed, installs if necessary, and returns the directory path.

    Args:
        install_dir: Override default install directory.

    Returns:
        Path to the BGMNwin directory.
    """
    installer = BGMNInstaller(install_dir=install_dir)
    if not installer.is_installed():
        installer.install()
    return installer.get_bgmn_dir()


def get_bgmn_path(install_dir: Path | None = None) -> Path:
    """Get path to the bgmn executable.

    Does NOT install automatically. Raises if not installed.

    Args:
        install_dir: Override default install directory.

    Returns:
        Path to the bgmn executable.

    Raises:
        BGMNNotInstalledError: If BGMN is not installed.
    """
    installer = BGMNInstaller(install_dir=install_dir)
    return installer.get_bgmn_path()
