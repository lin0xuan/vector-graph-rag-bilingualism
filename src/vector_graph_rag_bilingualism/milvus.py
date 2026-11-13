"""Optional Milvus mirror. Full payload/vector checks detect same-count mutations."""
import json
from .models import canonical, digest
from .storage import IntegrityError


def dump_collection(client, name):
    # Milvus Lite 2.5 can fail while projecting vectors from an empty segment.
    count = client.query(collection_name=name, filter="", output_fields=["count(*)"])
    if count and count[0].get("count(*)") == 0:
        return []
    iterator = client.query_iterator(collection_name=name, batch_size=500, output_fields=["*"])
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
