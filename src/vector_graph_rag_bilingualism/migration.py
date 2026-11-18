"""Explicit, model-free migration; source snapshots and collections are preserved."""
from copy import deepcopy
from pathlib import Path
import re
from .milvus import MilvusMirror, connect_standalone, standalone_uri, dump_collection, collection_hashes
from .models import Passage, digest
from .storage import SnapshotStore


def sync_chinese(config, snapshot=None):
    if config["language"] != "zh" or config.get("local_search"):
        raise ValueError("sync-milvus requires a Chinese Standalone profile")
    store = SnapshotStore(config.get("data_dir", ".data"))
    snapshot = snapshot or store.current(config["namespace"])
    manifest, records = store.load(snapshot)
    if manifest["language"] != "zh" or manifest["namespace"] != config["namespace"]:
        raise ValueError("Snapshot namespace/language mismatch")
    mirror = MilvusMirror.connect(config.get("milvus"))
    try:
        mirror.ensure(snapshot, manifest, records)
        return {"snapshot": snapshot, "collection": mirror.name(snapshot),
                "records": len(records), "status": "verified"}
    finally:
        mirror.client.close()


def migrate_english(config, source_data_dir, source_uri, snapshot):
    """Copy the pinned upstream's three standard collections, retaining vectors and IDs."""
    if config["language"] != "en" or not snapshot:
        raise ValueError("An English target profile and source snapshot are required")
    if Path(config.get("data_dir", ".data")).resolve() == Path(source_data_dir).resolve():
        raise ValueError("Use a separate target data_dir to preserve the source snapshot activation")
    prefix = config["upstream"].get("collection_prefix", "")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", prefix):
        raise ValueError("A nonempty valid target collection_prefix is required")
    source_store = SnapshotStore(source_data_dir)
    manifest, records = source_store.load(snapshot)
    if manifest["language"] != "en" or manifest["namespace"] != config["namespace"]:
        raise ValueError("Source snapshot namespace/language mismatch")
    from .config import UPSTREAM_COMMIT
    if manifest["config"]["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Source snapshot must use the pinned upstream version")
    settings = deepcopy(manifest["config"]["settings"])
    target_settings = config["upstream"]
    for key, value in target_settings.items():
        if key not in ("milvus_uri", "collection_prefix") and settings.get(key) != value:
            raise ValueError("Migration cannot change model/index settings; copy the source profile")
    source_names = [f'{settings["collection_prefix"]}_' + settings[k]
                    if settings.get("collection_prefix") else settings[k]
                    for k in ("entity_collection", "relation_collection", "passage_collection")]
    if set(source_names) != set(manifest["external"]["collections"]):
        raise ValueError("Source collection names do not match snapshot")
    settings["milvus_uri"] = standalone_uri(target_settings.get("milvus_uri"))
    settings["collection_prefix"] = target_settings["collection_prefix"]
    target_names = [f'{settings["collection_prefix"]}_{settings[k]}'
                    for k in ("entity_collection", "relation_collection", "passage_collection")]
    if not source_uri.startswith(("http://", "https://")) and not Path(source_uri).is_file():
        raise ValueError("Source Milvus Lite file does not exist")
    from pymilvus import MilvusClient, DataType
    source = MilvusClient(uri=source_uri, timeout=10)
    target = None
    created = []
    try:
        if collection_hashes(source, source_names) != manifest["external"]["collections"]:
            raise ValueError("Source Milvus contents differ from snapshot")
        target = connect_standalone({"uri": settings["milvus_uri"],
                                     "db_name": settings.get("milvus_db") or "default"})
        for old_name, new_name in zip(source_names, target_names):
            rows = dump_collection(source, old_name)
            expected = manifest["external"]["collections"][old_name]
            if target.has_collection(new_name):
                if digest(dump_collection(target, new_name)) != expected:
                    raise ValueError("Target collection already contains different data; choose a fresh prefix")
                continue
            schema = target.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=manifest["dimension"])
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            indexes = target.prepare_index_params()
            indexes.add_index(field_name="vector", index_type=settings["milvus_index_type"],
                              metric_type="IP", params=settings.get("milvus_index_params") or {})
            target.create_collection(collection_name=new_name, schema=schema, index_params=indexes,
                                     consistency_level="Strong")
            created.append(new_name)
            for start in range(0, len(rows), 100):
                target.insert(collection_name=new_name, data=rows[start:start + 100])
            target.flush(collection_name=new_name)
            if digest(dump_collection(target, new_name)) != expected:
                raise ValueError("Copied collection failed content verification")
        identity = {**manifest["config"], "settings": settings}
        target_store = SnapshotStore(config.get("data_dir", ".data"))
        new_snapshot = target_store.create(config["namespace"], "en",
            [Passage(**r["passage"]) for r in records], [r["vector"] for r in records], {}, identity,
            external={"collections": collection_hashes(target, target_names)})
        target_store.activate(config["namespace"], new_snapshot)
        return {"source_snapshot": snapshot, "snapshot": new_snapshot,
                "records": len(records), "status": "verified"}
    except Exception:
        # Only remove collections created by this attempt. Existing data is never replaced.
        if target is not None:
            for name in created:
                try:
                    target.drop_collection(name)
                except Exception:
                    pass  # Preserve the migration error if the server is unavailable during cleanup.
        raise
    finally:
        source.close()
        if target is not None:
            target.close()
