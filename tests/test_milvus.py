import pytest
from vector_graph_rag_bilingualism import ChinesePipeline, SnapshotStore, IntegrityError
from vector_graph_rag_bilingualism.milvus import MilvusMirror, dump_collection
from helpers import FakeEmbedding, FakeModel, corpus


def test_actual_milvus_query_and_vector_tamper(tmp_path, milvus_server_uri):
    pymilvus = pytest.importorskip("pymilvus")
    client = pymilvus.MilvusClient(uri=milvus_server_uri)
    snapshot = None
    try:
        mirror = MilvusMirror(client)
        import uuid
        pipeline = ChinesePipeline(SnapshotStore(tmp_path / "catalog"), "test_" + uuid.uuid4().hex,
                                   FakeEmbedding(), FakeModel(), mirror=mirror)
        snapshot = pipeline.build(corpus())
        assert pipeline.query("谁负责审批，期限多久？").status == "complete"
        name = mirror.name(snapshot)
        rows = dump_collection(client, name)
        rows[0]["vector"][0] += 0.25
        client.upsert(collection_name=name, data=[rows[0]])
        client.flush(name)
        with pytest.raises(IntegrityError, match="embedding"):
            pipeline.query("谁负责审批？")
    finally:
        if snapshot:
            client.drop_collection(mirror.name(snapshot))
        client.close()
