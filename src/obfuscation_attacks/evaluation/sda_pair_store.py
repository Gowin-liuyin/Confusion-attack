"""Temporary, disk-backed SDA training pairs with sequence-style access."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np


class SQLitePairStore:
    """Own a temporary database; close removes only the directory it created."""

    def __init__(self, parent: str | None = None):
        self.parent = parent
        self._tmp = None
        self._conn = None
        self._size = 0

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="sda-pairs-", dir=self.parent)
        try:
            self._conn = sqlite3.connect(str(Path(self._tmp.name) / "pairs.sqlite3"))
            self._conn.execute("PRAGMA journal_mode=OFF")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._conn.execute("CREATE TABLE pairs (i INTEGER PRIMARY KEY, src BLOB NOT NULL, tgt BLOB NOT NULL)")
            self._conn.execute("CREATE TABLE windows (doc INTEGER NOT NULL, start INTEGER NOT NULL, stop INTEGER NOT NULL, sha TEXT NOT NULL, draws INTEGER NOT NULL, PRIMARY KEY(doc, start, stop)) WITHOUT ROWID")
        except Exception:
            self.close()
            raise
        return self

    def append(self, pair):
        src, tgt = pair
        src_bytes = np.asarray(src, dtype="<i8").tobytes()
        tgt_bytes = np.asarray(tgt, dtype="<i8").tobytes()
        self._conn.execute("INSERT INTO pairs (i, src, tgt) VALUES (?, ?, ?)",
                           (self._size, src_bytes, tgt_bytes))
        self._size += 1

    def record_window(self, doc_index, start, stop, sha256):
        self._conn.execute(
            "INSERT INTO windows (doc,start,stop,sha,draws) VALUES (?,?,?,?,1) "
            "ON CONFLICT(doc,start,stop) DO UPDATE SET draws=draws+1",
            (int(doc_index), int(start), int(stop), sha256),
        )

    def publish_windows(self, path, *, reuse_existing=False, listing_cap=10):
        """Publish a full deterministic JSONL manifest without overwriting files."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        total, unique_content, draws = self._conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sha), COALESCE(SUM(draws), 0) FROM windows"
        ).fetchone()
        digest = hashlib.sha256()
        listed = []
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                for doc, start, stop, sha, count in self._conn.execute(
                    "SELECT doc,start,stop,sha,draws FROM windows ORDER BY doc,start,stop"
                ):
                    record = {"document_index": doc, "start": start, "end": stop,
                              "content_sha256": sha, "draws": count}
                    if len(listed) < listing_cap:
                        listed.append(record)
                    line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                    digest.update(line)
                    handle.write(line)
            expected = digest.hexdigest()
            if path.exists():
                if not reuse_existing:
                    raise FileExistsError(f"window manifest already exists: {path}")
                actual = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        actual.update(chunk)
                if actual.hexdigest() != expected:
                    raise ValueError("existing SDA window manifest does not match regenerated windows")
            else:
                # A hard link atomically publishes without replacing a concurrent file.
                os.link(temporary, path)
            return {"windows": listed, "listed": len(listed), "unique_windows": int(total),
                    "unique_content_hashes": int(unique_content), "draws": int(draws),
                    "manifest_path": str(path), "manifest_sha256": expected,
                    "independence_status": "unverified_overlapping_windows_are_not_independent"}
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def finish(self):
        self._conn.commit()

    def __len__(self):
        return self._size

    @staticmethod
    def _decode(src, tgt):
        return (np.frombuffer(src, dtype="<i8").tolist(),
                np.frombuffer(tgt, dtype="<i8").tolist())

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= self._size:
            raise IndexError(index)
        row = self._conn.execute("SELECT src, tgt FROM pairs WHERE i = ?", (index,)).fetchone()
        if row is None:
            raise RuntimeError("SDA pair store is incomplete")
        return self._decode(*row)

    def __iter__(self):
        cursor = self._conn.execute("SELECT src, tgt FROM pairs ORDER BY i")
        for row in cursor:
            yield self._decode(*row)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __exit__(self, exc_type, exc, tb):
        self.close()
