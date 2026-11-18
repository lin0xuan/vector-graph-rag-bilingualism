"""Standalone vector storage. Full payload/vector checks detect mutations."""
import json
import os
from urllib.parse import urlsplit
from .models import canonical, digest
from .storage import IntegrityError


DEFAULT_MILVUS_URI = "http://127.0.0.1:19530"


def standalone_uri(configured=None):
    """One server endpoint for both languages; never implicitly open Milvus Lite."""
    uri = os.getenv("VGRAG_MILVUS_URI") or configured or DEFAULT_MILVUS_URI
    parsed = urlsplit(uri)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("Milvus Standalone requires an http(s) server URI, not a .db file")
    # Validate malformed ports without including the URI in an error message.
    try:
        parsed.port
    except ValueError:
        raise ValueError("Invalid Milvus Standalone port") from None
    return uri.rstrip("/")


def connect_standalone(config=None):
    config = config or {}
    uri = standalone_uri(config.get("uri"))
    from pymilvus import MilvusClient
    try:
        return MilvusClient(uri=uri, token=os.getenv(config.get("token_env", "VGRAG_MILVUS_TOKEN"), ""),
                            db_name=config.get("db_name", "default"), timeout=10)
    except Exception:
        raise RuntimeError("Milvus Standalone connection failed; start docker compose up -d --wait") from None


def dump_collection(client, name):
    # Integrity checks must see recent writes, even on upstream Bounded collections.
    # Milvus Lite 2.5 can fail while projecting vectors from an empty segment.
    count = client.query(collection_name=name, filter="", output_fields=["count(*)"], consistency_level="Strong")
    if count and count[0].get("count(*)") == 0:
        return []
    iterator = client.query_iterator(collection_name=name, batch_size=500, output_fields=["*"], consistency_level="Strong")
    rows = []
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    # pymilvus may return numpy scalars; canonicalize the actual stored representation.
    def plain(value):
        if hasattr(value, "tolist"):
            return plain(value.tolist())
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    return sorted((plain(row) for row in rows), key=canonical)


def collection_hashes(client, names):
    return {name: digest(dump_collection(client, name)) for name in names}


class MilvusMirror:
    def __init__(self, client):
        self.client = client

    @staticmethod
    def name(snapshot):
        return "bi_" + snapshot[:40]

    @classmethod
    def connect(cls, config=None):
        return cls(connect_standalone(config))

    def ensure(self, snapshot, manifest, records):
        """Publish an old snapshot once, or verify an existing copy without replacing it."""
        if self.client.has_collection(self.name(snapshot)):
            self.validate(snapshot, records)
        else:
            self.publish(snapshot, manifest, records)

    def publish(self, snapshot, manifest, records):
        name = self.name(snapshot)
        if self.client.has_collection(name):
            raise ValueError("Snapshot collection already exists; never overwrite it")
        self.client.create_collection(collection_name=name, dimension=manifest["dimension"],
                                      primary_field_name="id", id_type="string", max_length=64,
                                      metric_type="COSINE", consistency_level="Strong")
        try:
            for start in range(0, len(records), 100):
                self.client.insert(collection_name=name, data=[
                    {"id": r["passage"]["id"], "vector": r["vector"], "payload": canonical(r)}
                    for r in records[start:start + 100]])
            self.client.flush(collection_name=name)
            self.validate(snapshot, records)
        except Exception:
            self.client.drop_collection(name)
            raise

    def validate(self, snapshot, records):
        if not self.client.has_collection(self.name(snapshot)):
            raise IntegrityError("Snapshot missing in Milvus; run sync-milvus before querying")
        rows = dump_collection(self.client, self.name(snapshot))
        expected = {r["passage"]["id"]: r for r in records}
        if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
            raise IntegrityError("Milvus record set changed")
        import struct
        for row in rows:
            target = expected[row["id"]]
            # Compare FLOAT_VECTOR at the storage precision, not Python's float64 precision.
            packed = lambda values: b"".join(struct.pack("<f", float(v)) for v in values)
            if json.loads(row["payload"]) != target or packed(row["vector"]) != packed(target["vector"]):
                raise IntegrityError("Milvus payload or embedding changed")

    def search(self, snapshot, query_vector, limit):
        rows = self.client.search(collection_name=self.name(snapshot), data=[query_vector],
                                  limit=limit, output_fields=["id"],
                                  search_params={"metric_type": "COSINE", "params": {}})
        return [row["id"] for row in rows[0]]
