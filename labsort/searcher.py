"""FTS5 search, filtered queries, duplicate detection."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .indexer import Index
from .utils import format_size, strip_copy_suffixes


def fts_search(
    index: Index,
    query: str,
    *,
    category: str | None = None,
    extension: str | None = None,
    directory: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Full-text search across filenames, paths, previews, and categories.

    Args:
        index: Open Index instance.
        query: FTS5 query string (supports AND, OR, NOT, "phrases").
        category: Filter by category name.
        extension: Filter by extension.
        directory: Filter to files under this directory.
        limit: Max results.
    """
    # FTS5 match
    params: list = []
    # Escape bare special chars for safety, but allow explicit FTS5 operators
    fts_query = query

    sql = """
        SELECT f.*,
               rank
        FROM files_fts fts
        JOIN files f ON f.rowid = fts.rowid
        WHERE files_fts MATCH ?
    """
    params.append(fts_query)

    if category:
        sql += " AND f.category = ?"
        params.append(category)
    if extension:
        sql += " AND f.extension = ?"
        params.append(extension)
    if directory:
        sql += " AND f.filepath LIKE ? || '%'"
        params.append(directory)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = index.conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def find_duplicates(index: Index, *, directory: str | None = None) -> list[dict]:
    """Find exact duplicates (same SHA256, different paths).

    Returns list of groups: {sha256, size, files: [{filepath, filename, ...}]}
    """
    sql = """
        SELECT sha256, size, COUNT(*) as cnt
        FROM files
        WHERE sha256 IS NOT NULL
    """
    params: list = []
    if directory:
        sql += " AND filepath LIKE ? || '%'"
        params.append(directory)
    sql += " GROUP BY sha256 HAVING cnt > 1 ORDER BY size DESC"

    dup_rows = index.conn.execute(sql, params).fetchall()
    groups = []
    for r in dup_rows:
        file_sql = "SELECT * FROM files WHERE sha256 = ?"
        file_params: list = [r["sha256"]]
        if directory:
            file_sql += " AND filepath LIKE ? || '%'"
            file_params.append(directory)
        files = index.conn.execute(file_sql, file_params).fetchall()
        groups.append({
            "sha256": r["sha256"],
            "size": r["size"],
            "count": r["cnt"],
            "files": [dict(f) for f in files],
        })
    return groups


def find_near_duplicates(
    index: Index,
    *,
    directory: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Find near-duplicates: files with the same base name after stripping copy suffixes.

    Returns groups: {base_name, files: [{filepath, filename, ...}]}
    """
    sql = "SELECT * FROM files"
    params: list = []
    if directory:
        sql += " WHERE filepath LIKE ? || '%'"
        params.append(directory)
    sql += f" LIMIT {limit * 10}"  # fetch more to find groups

    rows = index.conn.execute(sql, params).fetchall()

    # Group by (parent_dir, stripped_base_name)
    groups_map: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        f = dict(r)
        base = strip_copy_suffixes(f["filename"])
        groups_map[(f["parent_dir"], base)].append(f)

    groups = []
    for (pdir, base), files in groups_map.items():
        if len(files) > 1:
            groups.append({
                "base_name": base,
                "directory": pdir,
                "count": len(files),
                "files": files,
            })
    groups.sort(key=lambda g: g["count"], reverse=True)
    return groups[:limit]


def find_large_files(
    index: Index,
    *,
    min_size_mb: float = 100,
    directory: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Find files above a size threshold."""
    min_bytes = int(min_size_mb * 1024 * 1024)
    return index.query_files(
        min_size=min_bytes,
        parent_dir=directory or None,
        limit=limit,
    )


def find_recent_files(
    index: Index,
    *,
    days: int = 7,
    directory: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Find recently modified files."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    return index.query_files(
        modified_after=cutoff,
        parent_dir=directory or None,
        limit=limit,
    )


def find_orphaned_indices(index: Index) -> list[dict]:
    """Find index files (.bai, .tbi, etc.) whose data file is missing."""
    index_exts = (".bai", ".crai", ".tbi", ".csi", ".fai", ".dict")
    orphans = []
    for ext in index_exts:
        rows = index.conn.execute(
            "SELECT * FROM files WHERE extension = ?", (ext,)
        ).fetchall()
        for r in rows:
            f = dict(r)
            # Check if the corresponding data file exists in our index
            stem = Path(f["filename"]).stem
            data_check = index.conn.execute(
                "SELECT 1 FROM files WHERE parent_dir = ? AND filename LIKE ? AND extension != ? LIMIT 1",
                (f["parent_dir"], stem + "%", ext),
            ).fetchone()
            if not data_check:
                orphans.append(f)
    return orphans


def find_by_groups(index: Index, *, group_type: str | None = None) -> list[dict]:
    """Return file groups from the index."""
    if group_type:
        rows = index.conn.execute(
            "SELECT * FROM file_groups WHERE group_type = ?", (group_type,)
        ).fetchall()
    else:
        rows = index.conn.execute("SELECT * FROM file_groups").fetchall()
    results = []
    for r in rows:
        g = dict(r)
        g["members"] = json.loads(g["members"])
        results.append(g)
    return results
