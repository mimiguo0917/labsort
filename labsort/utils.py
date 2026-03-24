"""Shared helpers: size formatting, time parsing, hashing, etc."""

import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path

# Compound extensions recognized by labsort (order matters — longest first)
COMPOUND_EXTENSIONS = [
    ".fastq.gz", ".fq.gz",
    ".vcf.gz", ".vcf.bgz",
    ".bed.gz", ".gff.gz", ".gtf.gz",
    ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst",
    ".nii.gz",
    ".ome.tif", ".ome.tiff",
    ".scn.tif",
]


def get_compound_extension(filename: str) -> str:
    """Return the compound or simple extension (lowercased, with dot).

    Examples:
        'reads.fastq.gz' -> '.fastq.gz'
        'image.ome.tif'  -> '.ome.tif'
        'report.pdf'     -> '.pdf'
        'Makefile'       -> ''
    """
    lower = filename.lower()
    for cext in COMPOUND_EXTENSIONS:
        if lower.endswith(cext):
            return cext
    p = Path(filename)
    return p.suffix.lower()


def get_stem(filename: str) -> str:
    """Return the stem, stripping compound extensions."""
    ext = get_compound_extension(filename)
    if ext:
        return filename[: -len(ext)]
    return filename


def format_size(nbytes: int | float) -> str:
    """Human-readable file size."""
    if nbytes < 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            if unit == "B":
                return f"{int(nbytes)} B"
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def parse_timespan(s: str) -> timedelta:
    """Parse strings like '7d', '2w', '3m', '1y' into timedelta.

    Suffixes: s=seconds, m=minutes, h=hours, d=days, w=weeks, M=months(30d), y=years(365d)
    """
    m = re.fullmatch(r"(\d+)\s*([smhdwMy])", s.strip())
    if not m:
        raise ValueError(f"Invalid timespan: {s!r}. Use e.g. '7d', '2w', '3M', '1y'.")
    n = int(m.group(1))
    unit = m.group(2)
    mapping = {
        "s": timedelta(seconds=n),
        "m": timedelta(minutes=n),
        "h": timedelta(hours=n),
        "d": timedelta(days=n),
        "w": timedelta(weeks=n),
        "M": timedelta(days=30 * n),
        "y": timedelta(days=365 * n),
    }
    return mapping[unit]


def compute_sha256(filepath: Path, size_limit_mb: int = 500) -> str | None:
    """Compute SHA256 hex digest. Returns None if file exceeds size_limit_mb."""
    try:
        size = filepath.stat().st_size
        if size_limit_mb and size > size_limit_mb * 1024 * 1024:
            return None
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError):
        return None


def is_text_file(filepath: Path, sample_size: int = 8192) -> bool:
    """Heuristic: read first sample_size bytes, check for null bytes."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(sample_size)
        if b"\x00" in chunk:
            return False
        return True
    except (OSError, PermissionError):
        return False


def read_magic_bytes(filepath: Path, n: int = 16) -> bytes:
    """Read first n bytes of a file for magic-number detection."""
    try:
        with open(filepath, "rb") as f:
            return f.read(n)
    except (OSError, PermissionError):
        return b""


def content_preview(filepath: Path, max_chars: int = 500) -> str:
    """Read a short text preview for indexing. Returns '' for binary files."""
    if not is_text_file(filepath):
        return ""
    try:
        with open(filepath, "r", errors="replace") as f:
            return f.read(max_chars)
    except (OSError, PermissionError):
        return ""


def strip_copy_suffixes(name: str) -> str:
    """Strip common copy/version suffixes to find the 'base' name.

    'report (1).pdf'   -> 'report.pdf'
    'report_v2.pdf'    -> 'report.pdf'
    'report_copy.pdf'  -> 'report.pdf'
    'report - Copy.pdf' -> 'report.pdf'
    """
    stem = get_stem(name)
    ext = get_compound_extension(name)
    # Patterns to strip (order matters)
    patterns = [
        r"\s*\(\d+\)$",           # (1), (2)
        r"\s*-\s*Copy(\s*\(\d+\))?$",  # - Copy, - Copy (2)
        r"_copy\d*$",             # _copy, _copy2
        r"_v\d+$",               # _v2, _v10
        r"\s+\d+$",              # trailing space+number
    ]
    for pat in patterns:
        stem = re.sub(pat, "", stem, flags=re.IGNORECASE)
    return stem + ext


def iso_timestamp(dt: datetime | None = None) -> str:
    """Return ISO 8601 timestamp string."""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat(timespec="seconds")
