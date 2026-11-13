"""Immutable, content-checked SQLite snapshots with atomic activation."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import json
import sqlite3
from .models import Passage, canonical, digest, vector


class IntegrityError(ValueError):
    pass


class SnapshotStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "catalog.sqlite3"
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY, namespace TEXT, manifest TEXT);
                CREATE TABLE IF NOT EXISTS records(snapshot TEXT, id TEXT, payload TEXT,
                    PRIMARY KEY(snapshot,id));
                CREATE TABLE IF NOT EXISTS active(namespace TEXT PRIMARY KEY, snapshot TEXT);
                CREATE TABLE IF NOT EXISTS build_cache(key TEXT PRIMARY KEY, payload TEXT, checksum TEXT);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, namespace, language, passages, vectors, relations, config, *, external=None, telemetry=None):
        if language not in ("zh", "en") or not namespace:
            raise ValueError("Explicit namespace and language required")
        if not passages or len(passages) != len(vectors):
            raise ValueError("Passages and vectors must be nonempty and aligned")
        if len({p.id for p in passages}) != len(passages):
            raise ValueError("Duplicate passage IDs")
        if any(p.language != language for p in passages):
            raise ValueError("Mixed snapshot language")
        dimension = len(vectors[0])
        records = []
        known = {p.id for p in passages}
        for p, v in zip(passages, vectors):
            if not p.text or not p.source or any(n not in known for n in p.neighbors):
                raise ValueError("Invalid passage or neighbor")
            records.append({"passage": asdict(p), "vector": vector(v, dimension),
                            "relations": relations.get(p.id, [])})
        records.sort(key=lambda r: r["passage"]["id"])
        manifest = {"schema": 1, "namespace": namespace, "language": language,
                    "config": config, "dimension": dimension, "content_hash": digest(records),
                    "count": len(records), "external": external or {}, "telemetry": telemetry or []}
        snapshot_id = digest(manifest)
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?)", (snapshot_id, namespace, canonical(manifest)))
            for record in records:
                db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?)",
                           (snapshot_id, record["passage"]["id"], canonical(record)))
        self.load(snapshot_id)
        return snapshot_id

    def load(self, snapshot_id):
        with self.connection() as db:
            row = db.execute("SELECT namespace,manifest FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown snapshot")
            manifest = json.loads(row[1])
            records = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM records WHERE snapshot=? ORDER BY id", (snapshot_id,))]
        if digest(manifest) != snapshot_id or row[0] != manifest["namespace"]:
            raise IntegrityError("Snapshot manifest changed")
        if len(records) != manifest["count"] or digest(records) != manifest["content_hash"]:
            raise IntegrityError("Snapshot content changed")
        return manifest, records

    def activate(self, namespace, snapshot_id):
        manifest, _ = self.load(snapshot_id)
        if manifest["namespace"] != namespace:
            raise ValueError("Namespace mismatch")
        with self.connection() as db:
            db.execute("INSERT INTO active VALUES(?,?) ON CONFLICT(namespace) DO UPDATE SET snapshot=excluded.snapshot",
                       (namespace, snapshot_id))

    def current(self, namespace):
        with self.connection() as db:
            row = db.execute("SELECT snapshot FROM active WHERE namespace=?", (namespace,)).fetchone()
        if not row:
            raise KeyError("No active snapshot")
        return row[0]

    def cache_get(self, key):
        with self.connection() as db:
            row = db.execute("SELECT payload,checksum FROM build_cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if digest(value) != row[1]:
            raise IntegrityError("Build checkpoint changed")
        return value

    def cache_put(self, key, value):
        with self.connection() as db:
            db.execute("INSERT INTO build_cache VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload,checksum=excluded.checksum",
                       (key, canonical(value), digest(value)))
