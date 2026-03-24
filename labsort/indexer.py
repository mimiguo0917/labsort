"""SQLite + FTS5 database for file index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import ensure_config_dir, load_config
from .file_groups import FileGroup
from .scanner import FileRecord, ScanResult
from .utils import iso_timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    filepath TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL DEFAULT '',
    size INTEGER NOT NULL DEFAULT 0,
    created TEXT,
    modified TEXT,
    category TEXT NOT NULL DEFAULT 'other',
    sha256 TEXT,
    preview TEXT DEFAULT '',
    is_symlink INTEGER DEFAULT 0,
    parent_dir TEXT DEFAULT '',
    depth INTEGER DEFAULT 0,
    sidecar_of TEXT,
    indexed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_category ON files(category);
CREATE INDEX IF NOT EXISTS idx_files_extension ON files(extension);
CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256) WHERE sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_files_parent_dir ON files(parent_dir);
CREATE INDEX IF NOT EXISTS idx_files_modified ON files(modified);
CREATE INDEX IF NOT EXISTS idx_files_size ON files(size);

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    filepath, filename, category, preview, parent_dir,
    content='files',
    content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS file_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_type TEXT NOT NULL,
    primary_filepath TEXT NOT NULL,
    members TEXT NOT NULL,  -- JSON array of filepaths
    description TEXT DEFAULT '',
    scan_root TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS scan_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    file_count INTEGER DEFAULT 0,
    total_size INTEGER DEFAULT 0,
    skipped_unchanged INTEGER DEFAULT 0,
    skipped_ignored INTEGER DEFAULT 0,
    errors TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    op_type TEXT NOT NULL,       -- 'organize', 'clean', 'move'
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'pending',  -- 'pending', 'completed', 'undone', 'partial'
    file_count INTEGER DEFAULT 0,
    total_size INTEGER DEFAULT 0,
    source_root TEXT DEFAULT '',
    target_root TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS operation_moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id INTEGER NOT NULL REFERENCES operations(id),
    from_path TEXT NOT NULL,
    to_path TEXT NOT NULL,
    size INTEGER DEFAULT 0,
    category TEXT DEFAULT '',
    moved_at TEXT NOT NULL,
    undone INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_opmoves_opid ON operation_moves(operation_id);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, filepath, filename, category, preview, parent_dir)
    VALUES (new.rowid, new.filepath, new.filename, new.category, new.preview, new.parent_dir);
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, filepath, filename, category, preview, parent_dir)
    VALUES ('delete', old.rowid, old.filepath, old.filename, old.category, old.preview, old.parent_dir);
END;

CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, filepath, filename, category, preview, parent_dir)
    VALUES ('delete', old.rowid, old.filepath, old.filename, old.category, old.preview, old.parent_dir);
    INSERT INTO files_fts(rowid, filepath, filename, category, preview, parent_dir)
    VALUES (new.rowid, new.filepath, new.filename, new.category, new.preview, new.parent_dir);
END;
"""


class Index:
    """SQLite file index with FTS5 search."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            config = load_config()
            db_path = config["db_path"]
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Insert / update files
    # ------------------------------------------------------------------

    def upsert_files(self, records: list[FileRecord], batch_size: int = 500):
        """Insert or update file records in batches."""
        now = iso_timestamp()
        sql = """
            INSERT INTO files (filepath, filename, extension, size, created, modified,
                               category, sha256, preview, is_symlink, parent_dir, depth,
                               sidecar_of, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                filename=excluded.filename,
                extension=excluded.extension,
                size=excluded.size,
                created=excluded.created,
                modified=excluded.modified,
                category=excluded.category,
                sha256=excluded.sha256,
                preview=excluded.preview,
                is_symlink=excluded.is_symlink,
                parent_dir=excluded.parent_dir,
                depth=excluded.depth,
                sidecar_of=excluded.sidecar_of,
                indexed_at=excluded.indexed_at
        """
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            rows = [
                (
                    r.filepath, r.filename, r.extension, r.size, r.created,
                    r.modified, r.category, r.sha256, r.preview,
                    int(r.is_symlink), r.parent_dir, r.depth,
                    r.sidecar_of, now,
                )
                for r in batch
            ]
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def save_groups(self, groups: list[FileGroup], scan_root: str = ""):
        """Save file groups, replacing existing groups for the scan root."""
        if scan_root:
            self.conn.execute(
                "DELETE FROM file_groups WHERE scan_root = ?", (scan_root,)
            )
        for g in groups:
            self.conn.execute(
                """INSERT INTO file_groups (group_type, primary_filepath, members, description, scan_root)
                   VALUES (?, ?, ?, ?, ?)""",
                (g.group_type, g.primary, json.dumps(g.members), g.description, scan_root),
            )
        self.conn.commit()

    def log_scan(self, result: ScanResult):
        """Write a scan_log entry."""
        self.conn.execute(
            """INSERT INTO scan_log (root, started_at, finished_at, file_count,
                                     total_size, skipped_unchanged, skipped_ignored, errors)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.root,
                result.started_at,
                result.finished_at,
                len(result.files),
                result.total_size,
                result.skipped_unchanged,
                result.skipped_ignored,
                json.dumps(result.errors),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_last_indexed(self, root: str) -> dict[str, str]:
        """Get {filepath: modified} for all files under root, for incremental scan."""
        rows = self.conn.execute(
            "SELECT filepath, modified FROM files WHERE filepath LIKE ? || '%'",
            (root,),
        ).fetchall()
        return {r["filepath"]: r["modified"] for r in rows}

    def get_file(self, filepath: str) -> dict | None:
        """Get a single file record."""
        row = self.conn.execute(
            "SELECT * FROM files WHERE filepath = ?", (filepath,)
        ).fetchone()
        return dict(row) if row else None

    def get_category_counts(self) -> list[tuple[str, int, int]]:
        """Return [(category, count, total_size)] sorted by count desc."""
        rows = self.conn.execute(
            "SELECT category, COUNT(*) as cnt, SUM(size) as total "
            "FROM files GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        return [(r["category"], r["cnt"], r["total"]) for r in rows]

    def get_extension_counts(self) -> list[tuple[str, int]]:
        """Return [(extension, count)] sorted by count desc."""
        rows = self.conn.execute(
            "SELECT extension, COUNT(*) as cnt FROM files GROUP BY extension ORDER BY cnt DESC"
        ).fetchall()
        return [(r["extension"], r["cnt"]) for r in rows]

    def total_files(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as c FROM files").fetchone()
        return row["c"]

    def total_size(self) -> int:
        row = self.conn.execute("SELECT COALESCE(SUM(size), 0) as s FROM files").fetchone()
        return row["s"]

    def get_scan_roots(self) -> list[str]:
        """Return distinct scan roots."""
        rows = self.conn.execute(
            "SELECT DISTINCT root FROM scan_log ORDER BY root"
        ).fetchall()
        return [r["root"] for r in rows]

    def last_scan(self) -> dict | None:
        """Return the most recent scan log entry."""
        row = self.conn.execute(
            "SELECT * FROM scan_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_groups(self, scan_root: str | None = None) -> list[dict]:
        """Return file groups, optionally filtered by scan root."""
        if scan_root:
            rows = self.conn.execute(
                "SELECT * FROM file_groups WHERE scan_root = ?", (scan_root,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM file_groups").fetchall()
        return [dict(r) for r in rows]

    def rebuild_fts(self):
        """Rebuild the FTS5 index from scratch."""
        self.conn.execute("INSERT INTO files_fts(files_fts) VALUES('rebuild')")
        self.conn.commit()

    def remove_missing_files(self) -> int:
        """Remove files from the index that no longer exist on disk. Returns count removed."""
        rows = self.conn.execute("SELECT filepath FROM files").fetchall()
        missing = [r["filepath"] for r in rows if not Path(r["filepath"]).exists()]
        if missing:
            for i in range(0, len(missing), 500):
                batch = missing[i : i + 500]
                placeholders = ",".join("?" * len(batch))
                self.conn.execute(
                    f"DELETE FROM files WHERE filepath IN ({placeholders})", batch
                )
            self.conn.commit()
        return len(missing)

    def query_files(self, **filters) -> list[dict]:
        """Query files with optional filters.

        Supported filters: category, extension, parent_dir, min_size, max_size,
                          modified_after, modified_before, limit.
        """
        clauses = []
        params: list = []

        if filters.get("category"):
            clauses.append("category = ?")
            params.append(filters["category"])
        if filters.get("extension"):
            clauses.append("extension = ?")
            params.append(filters["extension"])
        if filters.get("parent_dir"):
            clauses.append("parent_dir LIKE ? || '%'")
            params.append(filters["parent_dir"])
        if filters.get("min_size") is not None:
            clauses.append("size >= ?")
            params.append(filters["min_size"])
        if filters.get("max_size") is not None:
            clauses.append("size <= ?")
            params.append(filters["max_size"])
        if filters.get("modified_after"):
            clauses.append("modified >= ?")
            params.append(filters["modified_after"])
        if filters.get("modified_before"):
            clauses.append("modified <= ?")
            params.append(filters["modified_before"])

        where = " AND ".join(clauses) if clauses else "1=1"
        limit = filters.get("limit", 100)
        sql = f"SELECT * FROM files WHERE {where} ORDER BY modified DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Operations (Phase 2)
    # ------------------------------------------------------------------

    def create_operation(self, op_type: str, description: str = "",
                         source_root: str = "", target_root: str = "") -> int:
        """Create a new operation record and return its ID."""
        now = iso_timestamp()
        cur = self.conn.execute(
            """INSERT INTO operations (op_type, description, created_at, status,
                                       source_root, target_root)
               VALUES (?, ?, ?, 'pending', ?, ?)""",
            (op_type, description, now, source_root, target_root),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_move(self, operation_id: int, from_path: str, to_path: str,
                 size: int = 0, category: str = ""):
        """Log a single file move within an operation."""
        now = iso_timestamp()
        self.conn.execute(
            """INSERT INTO operation_moves (operation_id, from_path, to_path,
                                            size, category, moved_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (operation_id, from_path, to_path, size, category, now),
        )

    def complete_operation(self, operation_id: int, file_count: int, total_size: int):
        """Mark an operation as completed."""
        now = iso_timestamp()
        self.conn.execute(
            """UPDATE operations SET status='completed', completed_at=?,
                                     file_count=?, total_size=?
               WHERE id=?""",
            (now, file_count, total_size, operation_id),
        )
        self.conn.commit()

    def get_operation(self, operation_id: int) -> dict | None:
        """Get an operation by ID."""
        row = self.conn.execute(
            "SELECT * FROM operations WHERE id = ?", (operation_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_operation_moves(self, operation_id: int) -> list[dict]:
        """Get all moves for an operation."""
        rows = self.conn.execute(
            "SELECT * FROM operation_moves WHERE operation_id = ? ORDER BY id",
            (operation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_operations(self, limit: int = 20) -> list[dict]:
        """Get recent operations."""
        rows = self.conn.execute(
            "SELECT * FROM operations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_operation_undone(self, operation_id: int):
        """Mark an operation and all its moves as undone."""
        self.conn.execute(
            "UPDATE operations SET status='undone' WHERE id=?", (operation_id,)
        )
        self.conn.execute(
            "UPDATE operation_moves SET undone=1 WHERE operation_id=?", (operation_id,)
        )
        self.conn.commit()

    def update_file_path(self, old_path: str, new_path: str):
        """Update a file's path in the index after a move."""
        new_parent = str(Path(new_path).parent)
        new_filename = Path(new_path).name
        self.conn.execute(
            """UPDATE files SET filepath=?, filename=?, parent_dir=?
               WHERE filepath=?""",
            (new_path, new_filename, new_parent, old_path),
        )

    def batch_update_paths(self, moves: list[tuple[str, str]]):
        """Update multiple file paths in the index. moves = [(old_path, new_path), ...]."""
        for old_path, new_path in moves:
            self.update_file_path(old_path, new_path)
        self.conn.commit()
