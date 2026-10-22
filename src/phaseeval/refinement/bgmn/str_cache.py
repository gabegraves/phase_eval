"""
Hash-based .str file cache with age-based cleanup.

Caches generated BGMN .str files by content hash to avoid redundant
CIF-to-STR conversions across runs. Cache key is SHA256 of
(CIF content + lebail flag + sorted params dict).

Cache structure: {cache_dir}/{hash[:2]}/{hash}.str (two-level for
filesystem friendliness on large caches).

Thread safety: atomic writes (write to tmp then rename).

References:
- BGMN manual: .str file format
- src/phaseeval/refinement/bgmn/str_generator.py: cif_to_str converter
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import UTC
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".axa" / "cache" / "str"
# v3: TDS unit fix (AXA-ynoj) — B_iso now converted Angstrom^2 -> nm^2 at the write
# boundary. The generator's code is not part of the cache key, so every generator
# behavior change MUST bump this version or pre-fix .str files are served verbatim.
_STR_CACHE_VERSION = 3


class StrCache:
    """Hash-based cache for BGMN .str structure files.

    Avoids redundant CIF-to-STR conversions by caching generated .str files
    keyed by (CIF content + generation parameters).

    Cache layout::

        {cache_dir}/
            ab/
                abcdef1234567890.str
            cd/
                cdef0123456789ab.str

    Example::

        cache = StrCache(cache_dir=Path("/tmp/str_cache"))
        result = cache.get_or_generate(cif_path, output_dir, lebail=True)
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the STR cache.

        Args:
            cache_dir: Directory for cached .str files.
                Defaults to ``~/.axa/cache/str/``.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, cif_path: Path, lebail: bool, **params: Any) -> str:
        """Compute SHA256 cache key from CIF content and parameters.

        Args:
            cif_path: Path to the CIF file (content is hashed).
            lebail: Le Bail mode flag.
            **params: Additional generation parameters.

        Returns:
            Hex string of SHA256 hash.
        """
        cif_content = cif_path.read_text(encoding="utf-8", errors="replace")

        key_data = {
            "cif_content": cif_content,
            "lebail": lebail,
            "params": dict(sorted(params.items())),
            "format_version": _STR_CACHE_VERSION,
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Get the filesystem path for a cache key.

        Two-level directory structure: {hash[:2]}/{hash}.str
        """
        return self.cache_dir / key[:2] / f"{key}.str"

    def get(self, cif_path: Path, lebail: bool = True, **params: Any) -> Path | None:
        """Look up a cached .str file.

        Args:
            cif_path: Path to the CIF file.
            lebail: Le Bail mode flag.
            **params: Generation parameters.

        Returns:
            Path to cached .str file if found, None otherwise.
        """
        key = self._cache_key(cif_path, lebail, **params)
        cached = self._cache_path(key)
        if cached.exists():
            logger.debug(f"Cache hit for {cif_path.name}: {key[:12]}...")
            return cached
        logger.debug(f"Cache miss for {cif_path.name}: {key[:12]}...")
        return None

    def put(self, cif_path: Path, str_path: Path, lebail: bool = True, **params: Any) -> Path:
        """Store a .str file in the cache.

        Uses atomic write (tmp file + rename) for thread safety.

        Args:
            cif_path: Path to the source CIF file.
            str_path: Path to the generated .str file to cache.
            lebail: Le Bail mode flag.
            **params: Generation parameters.

        Returns:
            Path to the cached copy.
        """
        key = self._cache_key(cif_path, lebail, **params)
        cached = self._cache_path(key)
        cached.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=cached.parent, suffix=".str.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(str_path.read_text())
            os.replace(tmp_path, cached)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.debug(f"Cached {cif_path.name} -> {key[:12]}...")
        return cached

    def get_or_generate(
        self,
        cif_path: Path,
        output_dir: Path,
        lebail: bool = True,
        **params: Any,
    ) -> Path:
        """Get from cache or generate and cache a .str file.

        Args:
            cif_path: Path to the CIF file.
            output_dir: Directory for generated .str file (if cache miss).
            lebail: Le Bail mode flag.
            **params: Additional generation parameters passed to cif_to_str.

        Returns:
            Path to the .str file (either cached or freshly generated).
        """
        cached = self.get(cif_path, lebail, **params)
        if cached is not None:
            return cached

        # Generate fresh
        from phaseeval.refinement.bgmn.str_generator import cif_to_str

        str_path = cif_to_str(cif_path, output_dir, lebail=lebail, **params)

        # Cache the result
        cached = self.put(cif_path, str_path, lebail, **params)
        return cached

    def cleanup(self, max_age_days: int = 30, max_size_mb: int = 500) -> int:
        """Remove old cached files.

        Removes files older than ``max_age_days`` first, then if total
        size exceeds ``max_size_mb``, removes oldest files until under limit.

        Args:
            max_age_days: Maximum age in days for cached files.
            max_size_mb: Maximum total cache size in megabytes.

        Returns:
            Number of files removed.
        """
        removed = 0
        now = time.time()
        max_age_seconds = max_age_days * 86400
        max_size_bytes = max_size_mb * 1024 * 1024

        # Collect all .str files with their metadata
        files: list[tuple[Path, float, int]] = []  # (path, mtime, size)
        for str_file in self.cache_dir.rglob("*.str"):
            stat = str_file.stat()
            files.append((str_file, stat.st_mtime, stat.st_size))

        # Phase 1: Remove files older than max_age_days
        remaining = []
        for path, mtime, size in files:
            if (now - mtime) > max_age_seconds:
                path.unlink()
                removed += 1
                logger.debug(f"Removed expired cache file: {path.name}")
            else:
                remaining.append((path, mtime, size))

        # Phase 2: If still over size limit, remove oldest first
        total_size = sum(s for _, _, s in remaining)
        if total_size > max_size_bytes:
            remaining.sort(key=lambda x: x[1])  # oldest first
            for path, _mtime, size in remaining:
                if total_size <= max_size_bytes:
                    break
                path.unlink()
                total_size -= size
                removed += 1
                logger.debug(f"Removed oversized cache file: {path.name}")

        # Clean up empty subdirectories
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()

        logger.info(f"Cache cleanup: removed {removed} files")
        return removed

    def clear(self) -> int:
        """Remove all cached .str files.

        Returns:
            Number of files removed.
        """
        removed = 0
        for str_file in self.cache_dir.rglob("*.str"):
            str_file.unlink()
            removed += 1

        # Clean up empty subdirectories
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()

        logger.info(f"Cache cleared: removed {removed} files")
        return removed

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with:
            - file_count: Number of cached .str files
            - total_size_mb: Total cache size in megabytes
            - oldest_entry: ISO timestamp of oldest file (or None)
            - newest_entry: ISO timestamp of newest file (or None)
        """
        files = list(self.cache_dir.rglob("*.str"))

        if not files:
            return {
                "file_count": 0,
                "total_size_mb": 0.0,
                "oldest_entry": None,
                "newest_entry": None,
            }

        total_size = 0
        oldest_mtime = float("inf")
        newest_mtime = 0.0

        for f in files:
            stat = f.stat()
            total_size += stat.st_size
            oldest_mtime = min(oldest_mtime, stat.st_mtime)
            newest_mtime = max(newest_mtime, stat.st_mtime)

        from datetime import datetime

        return {
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 3),
            "oldest_entry": datetime.fromtimestamp(oldest_mtime, tz=UTC).isoformat(),
            "newest_entry": datetime.fromtimestamp(newest_mtime, tz=UTC).isoformat(),
        }
