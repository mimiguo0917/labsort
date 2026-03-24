"""File discovery, metadata collection, and incremental scanning."""

from __future__ import annotations

import fnmatch
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from .classifier import classify
from .config import load_config
from .utils import (
    compute_sha256,
    content_preview,
    format_size,
    get_compound_extension,
    iso_timestamp,
    read_magic_bytes,
)


@dataclass
class FileRecord:
    """All metadata collected for a single file."""
    filepath: str
    filename: str
    extension: str
    size: int
    created: str
    modified: str
    category: str
    sha256: str | None = None
    preview: str = ""
    magic_bytes: bytes = b""
    is_symlink: bool = False
    parent_dir: str = ""
    depth: int = 0
    sidecar_of: str | None = None


@dataclass
class ScanResult:
    """Results from a scan operation."""
    files: list[FileRecord] = field(default_factory=list)
    root: str = ""
    started_at: str = ""
    finished_at: str = ""
    total_size: int = 0
    skipped_unchanged: int = 0
    skipped_ignored: int = 0
    errors: list[str] = field(default_factory=list)


def _should_ignore(name: str, path: Path, config: dict, is_dir: bool) -> bool:
    """Check if a file/dir should be skipped based on config rules."""
    if is_dir:
        if name in config["ignore_dirs"]:
            return True
        for pattern in config.get("ignore_dir_patterns", []):
            if fnmatch.fnmatch(name, pattern):
                return True
        if not config["scan_hidden"] and name.startswith("."):
            return True
    else:
        for pattern in config["ignore_patterns"]:
            if fnmatch.fnmatch(name, pattern):
                return True
        if not config["scan_hidden"] and name.startswith("."):
            return True
    return False


def _get_siblings_for_dir(dirpath: Path) -> dict[str, list[str]]:
    """Pre-fetch all filenames in a directory, keyed by filename for O(1) lookup.

    Returns a dict where every file maps to its sibling list.
    """
    try:
        names = [e.name for e in os.scandir(dirpath) if e.is_file()]
    except (OSError, PermissionError):
        names = []
    sibling_map = {}
    for n in names:
        sibling_map[n] = [s for s in names if s != n]
    return sibling_map


SIDECAR_EXTENSIONS = {".json", ".yaml", ".yml", ".xml"}


def _find_sidecar_target(filepath: Path, siblings: list[str]) -> str | None:
    """If this file is a sidecar (.json/.yaml companion), return the primary file it belongs to."""
    ext = filepath.suffix.lower()
    if ext not in SIDECAR_EXTENSIONS:
        return None
    stem = filepath.stem
    for sib in siblings:
        sib_stem = Path(sib).stem
        sib_ext = Path(sib).suffix.lower()
        if sib_stem == stem and sib_ext not in SIDECAR_EXTENSIONS:
            return sib
    return None


def scan_directory(
    root: str | Path,
    *,
    thorough: bool = False,
    dry_run: bool = False,
    last_indexed: dict[str, str] | None = None,
    show_progress: bool = True,
) -> ScanResult:
    """Scan a directory tree and collect file metadata.

    Args:
        root: Directory to scan.
        thorough: If True, hash files even above the size limit.
        dry_run: If True, only count files without collecting full metadata.
        last_indexed: Dict of {filepath: mtime_iso} for incremental scanning.
        show_progress: Show Rich progress bar.

    Returns:
        ScanResult with all discovered FileRecord objects.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    config = load_config()
    hash_limit = 0 if thorough else config["hash_size_limit_mb"]
    last_indexed = last_indexed or {}

    result = ScanResult(root=str(root), started_at=iso_timestamp())
    visited_inodes: set[int] = set()

    # First pass: count files for progress bar
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dp = Path(dirpath)
        # Prune ignored dirs in-place
        dirnames[:] = [
            d for d in dirnames
            if not _should_ignore(d, dp / d, config, is_dir=True)
        ]
        for fn in filenames:
            if not _should_ignore(fn, dp / fn, config, is_dir=False):
                file_count += 1

    progress_ctx = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        disable=not show_progress,
    )

    with progress_ctx as progress:
        task = progress.add_task("Scanning", total=file_count)

        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            dp = Path(dirpath)

            # Symlink loop detection
            try:
                dir_stat = dp.stat()
                inode = dir_stat.st_ino
                if inode in visited_inodes:
                    result.errors.append(f"Symlink loop detected: {dp}")
                    dirnames.clear()
                    continue
                visited_inodes.add(inode)
            except OSError:
                dirnames.clear()
                continue

            # Prune ignored dirs
            dirnames[:] = [
                d for d in dirnames
                if not _should_ignore(d, dp / d, config, is_dir=True)
            ]

            # Pre-fetch siblings for classification
            sibling_map = _get_siblings_for_dir(dp)

            for fn in filenames:
                if _should_ignore(fn, dp / fn, config, is_dir=False):
                    result.skipped_ignored += 1
                    progress.advance(task)
                    continue

                fp = dp / fn

                # Resolve symlinks, skip if broken
                is_symlink = fp.is_symlink()
                try:
                    fstat = fp.stat()
                except OSError as e:
                    result.errors.append(f"Cannot stat {fp}: {e}")
                    progress.advance(task)
                    continue

                # Skip non-regular files
                if not stat.S_ISREG(fstat.st_mode):
                    progress.advance(task)
                    continue

                filepath_str = str(fp)
                mtime = datetime.fromtimestamp(fstat.st_mtime).isoformat(timespec="seconds")

                # Incremental: skip if unchanged
                if filepath_str in last_indexed and last_indexed[filepath_str] == mtime:
                    result.skipped_unchanged += 1
                    progress.advance(task)
                    continue

                if dry_run:
                    result.total_size += fstat.st_size
                    result.files.append(FileRecord(
                        filepath=filepath_str,
                        filename=fn,
                        extension=get_compound_extension(fn),
                        size=fstat.st_size,
                        created="",
                        modified=mtime,
                        category="",
                        parent_dir=str(dp),
                    ))
                    progress.advance(task)
                    continue

                # Full metadata collection
                try:
                    ctime = datetime.fromtimestamp(fstat.st_birthtime).isoformat(timespec="seconds")
                except AttributeError:
                    ctime = datetime.fromtimestamp(fstat.st_ctime).isoformat(timespec="seconds")

                ext = get_compound_extension(fn)
                siblings = sibling_map.get(fn, [])
                category = classify(fp, siblings=siblings)
                sha = compute_sha256(fp, size_limit_mb=hash_limit)
                preview = content_preview(fp) if fstat.st_size < 1024 * 1024 else ""
                depth = len(fp.relative_to(root).parts) - 1
                sidecar_target = _find_sidecar_target(fp, siblings)

                rec = FileRecord(
                    filepath=filepath_str,
                    filename=fn,
                    extension=ext,
                    size=fstat.st_size,
                    created=ctime,
                    modified=mtime,
                    category=category,
                    sha256=sha,
                    preview=preview,
                    is_symlink=is_symlink,
                    parent_dir=str(dp),
                    depth=depth,
                    sidecar_of=sidecar_target,
                )
                result.files.append(rec)
                result.total_size += fstat.st_size
                progress.advance(task)

    result.finished_at = iso_timestamp()
    return result
