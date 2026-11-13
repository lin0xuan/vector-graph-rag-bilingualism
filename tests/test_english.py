from unittest.mock import patch
import pytest
from vector_graph_rag_bilingualism import EnglishUpstreamPipeline, SnapshotStore, Passage, IntegrityError
from vector_graph_rag_bilingualism.english import verify_upstream_version
from vector_graph_rag_bilingualism.milvus import dump_collection
from helpers import FakeEmbedding


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    pytest.importorskip("vector_graph_rag")
    from vector_graph_rag import VectorGraphRAG, Settings
    from vector_graph_rag.llm.extractor import EntityExtractor
    from vector_graph_rag.llm.jev import JevReranker
    from vector_graph_rag.llm.reranker import AnswerGenerator
    import tiktoken
    class Tokens:
        def encode(self, text, **kwargs):
            return list(text)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: Tokens())
    monkeypatch.setattr(EntityExtractor, "extract", lambda self, question: ["Alice"])
    monkeypatch.setattr(AnswerGenerator, "generate", lambda self, question, passages: " | ".join(passages))
    def scores(self, payloads):
        result = {}
        for payload in payloads:
            result.update({key: 0.8 for key in payload["questions"]})
        return result
    monkeypatch.setattr(JevReranker, "_score_payloads", scores)
    embedding = FakeEmbedding()
    # Native EmbeddingModel.embed takes one string, unlike our batch adapter.
    class NativeEmbedding:
        dimension = embedding.dimension
        def embed(self, text, **kwargs):
            return embedding.embed([text])[0]
        def embed_batch(self, texts, **kwargs):
            return embedding.embed(texts)
    settings = Settings(milvus_uri=str(tmp_path / "english.db"), collection_prefix="english_fixture",
                        openai_api_key="offline-test", jev_api_key="offline-test", reranker_model="jev-1.13.0",
                        llm_model="fixture-chat", embedding_model="fixture-embedding", embedding_provider="openai",
                        embedding_dimension=embedding.dimension, final_top_k=3,
                        entity_similarity_threshold=-1.0, use_llm_cache=False)
    with patch("vector_graph_rag.rag.EmbeddingModel", return_value=NativeEmbedding()):
        rag = VectorGraphRAG(settings=settings)
    try:
        yield rag
    finally:
        rag._store.client.close()


def test_native_upstream_and_adapter_share_results_and_index(tmp_path, upstream):
    verify_upstream_version()
    passages = [Passage("p1", "people", "Alice leads Project A.", "en", "people"),
                Passage("p2", "location", "Project A is based in London.", "en", "location")]
    adapter = EnglishUpstreamPipeline(upstream, SnapshotStore(tmp_path / "catalog"), "en")
    snapshot = adapter.build(passages, extract_triplets=False,
                            triplets={"p1": [["Alice", "leads", "Project A"]],
                                      "p2": [["Project A", "is based in", "London"]]})
    native = upstream.query("Where is Alice's project based?")
    result = adapter.query("Where is Alice's project based?")
    retrieved = adapter.retrieve("Where is Alice's project based?")
    assert result.answer == native.answer
    assert [p.text for p in result.passages] == native.passages
    assert retrieved.snapshot_id == result.snapshot_id == snapshot
    assert retrieved.answer == ""
    assert [p.text for p in retrieved.passages] == [p.text for p in result.passages]
    filtered = adapter.retrieve("Where is Alice?", filter='source == "people"')
    assert {p.source for p in filtered.passages} <= {"people"}
    # A same-count upstream vector edit must fail before native retrieval runs.
    rows = dump_collection(adapter.client, adapter.passage_collection)
    rows[0]["vector"][0] += 0.2
    adapter.client.upsert(collection_name=adapter.passage_collection, data=[rows[0]])
    adapter.client.flush(adapter.passage_collection)
    with pytest.raises(IntegrityError, match="Milvus"):
        adapter.query("Where is Alice?")


def test_duplicate_text_keeps_source_aliases_without_claiming_unique_citation(tmp_path, upstream):
    adapter = EnglishUpstreamPipeline(upstream, SnapshotStore(tmp_path / "catalog"), "en")
    passages = [Passage("p1", "one", "Alice leads Project A.", "en", "one"),
                Passage("p2", "two", "Alice leads Project A.", "en", "two")]
    adapter.build(passages, extract_triplets=False,
                  triplets={p.id: [["Alice", "leads", "Project A"]] for p in passages})
    result = adapter.query("Who leads Project A?")
    assert {p.source for p in result.passages} == {"one", "two"}
    assert result.status == "partial"
    assert result.trace.events[0]["ambiguous_citations"]
    filtered = adapter.query("Who leads Project A?", filter='source == "one"')
    assert [p.source for p in filtered.passages] == ["one"]
    assert filtered.status == "complete"
    with pytest.raises(ValueError, match="fresh"):
        adapter.build(passages, extract_triplets=False)


def test_native_failure_propagates_instead_of_empty_success(tmp_path, upstream, monkeypatch):
    adapter = EnglishUpstreamPipeline(upstream, SnapshotStore(tmp_path / "catalog"), "en")
    adapter.build([Passage("p1", "one", "Alice leads Project A.", "en", "one")], extract_triplets=False,
                  triplets={"p1": [["Alice", "leads", "Project A"]]})
    def failure(*args, **kwargs):
        raise TimeoutError("fixture")
    monkeypatch.setattr(upstream._reranker, "_score_payloads", failure)
    with pytest.raises(TimeoutError):
        adapter.query("Who leads Project A?")
