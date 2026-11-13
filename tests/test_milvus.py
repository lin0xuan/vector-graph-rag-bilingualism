import pytest
from vector_graph_rag_bilingualism import ChinesePipeline, SnapshotStore, IntegrityError
from vector_graph_rag_bilingualism.milvus import MilvusMirror, dump_collection
from helpers import FakeEmbedding, FakeModel, corpus


def test_actual_milvus_query_and_vector_tamper(tmp_path):
    pymilvus = pytest.importorskip("pymilvus")
    client = pymilvus.MilvusClient(uri=str(tmp_path / "milvus.db"))
    try:
        mirror = MilvusMirror(client)
        pipeline = ChinesePipeline(SnapshotStore(tmp_path / "catalog"), "zh", FakeEmbedding(), FakeModel(), mirror=mirror)
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
        client.close()
