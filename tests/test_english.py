from unittest.mock import Mock, patch
from types import SimpleNamespace
import json
import pytest
from vector_graph_rag_bilingualism import EnglishUpstreamPipeline, SnapshotStore, Passage, IntegrityError
from vector_graph_rag_bilingualism.config import EnglishQueryConfig
from vector_graph_rag_bilingualism.english import verify_upstream_version
from vector_graph_rag_bilingualism.milvus import dump_collection
from helpers import FakeEmbedding


@pytest.fixture
def upstream(tmp_path, monkeypatch, request):
    pytest.importorskip("vector_graph_rag")
    from vector_graph_rag import VectorGraphRAG, Settings
    from vector_graph_rag.llm.extractor import EntityExtractor
    from vector_graph_rag.llm.jev import JevReranker
    from vector_graph_rag.llm.reranker import AnswerGenerator, LLMReranker
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
    monkeypatch.setattr(LLMReranker, "rerank", lambda self, question, ids, texts: (ids, texts))
    embedding = FakeEmbedding()
    # Native EmbeddingModel.embed takes one string, unlike our batch adapter.
    class NativeEmbedding:
        dimension = embedding.dimension
        def embed(self, text, **kwargs):
            return embedding.embed([text])[0]
        def embed_batch(self, texts, **kwargs):
            return embedding.embed(texts)
    settings = Settings(milvus_uri=str(tmp_path / "english.db"), collection_prefix="english_fixture",
                        openai_api_key="offline-test", jev_api_key="offline-test",
                        reranker_model=None if getattr(request, "param", "jev") == "llm" else "jev-1.13.0",
                        llm_model="fixture-chat", embedding_model="fixture-embedding", embedding_provider="openai",
                        embedding_dimension=embedding.dimension, final_top_k=3,
                        entity_similarity_threshold=-1.0, use_llm_cache=False)
    with patch("vector_graph_rag.rag.EmbeddingModel", return_value=NativeEmbedding()):
        rag = VectorGraphRAG(settings=settings)
    if not settings.uses_jev:
        client = Mock(base_url="https://api.siliconflow.cn/v1")
        client.with_options.return_value = client
        def completion(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            ids = [x["id"] for x in payload["candidates"]][:payload["count"]]
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps({"ranked_ids": ids})))])
        client.chat.completions.create.side_effect = completion
        rag._reranker.client = client
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


def test_jev_failure_falls_back_to_naive_query(tmp_path, upstream, monkeypatch):
    adapter = EnglishUpstreamPipeline(upstream, SnapshotStore(tmp_path / "catalog"), "en")
    adapter.build([Passage("p1", "one", "Alice leads Project A.", "en", "one")], extract_triplets=False,
                  triplets={"p1": [["Alice", "leads", "Project A"]]})
    def failure(*args, **kwargs):
        raise TimeoutError("fixture")
    monkeypatch.setattr(upstream._reranker, "_score_payloads", failure)
    result = adapter.query("Who leads Project A?")
    assert result.answer == "Alice leads Project A."
    assert result.trace.events[0]["status"] == "fallback"


@pytest.mark.parametrize("upstream", ["llm"], indirect=True)
@pytest.mark.parametrize("graph_count", [0, 1, 3, 4])
def test_query_and_retrieve_share_merged_passage_ranking(
        tmp_path, upstream, monkeypatch, graph_count):
    adapter = EnglishUpstreamPipeline(upstream, SnapshotStore(tmp_path / "catalog"), "en")
    passages = [Passage(f"p{i}", f"source{i}", f"Alice leads Project {i}.", "en", f"source{i}")
                for i in range(4)]
    adapter.build(passages, extract_triplets=False,
                  triplets={p.id: [["Alice", "leads", f"Project {i}"]] for i, p in enumerate(passages)})
    # Control relation selection size while keeping real Milvus dense retrieval.
    selected = passages[:graph_count]
    monkeypatch.setattr(upstream, "_get_passages_from_relations",
                        lambda ids, filter=None, round_robin=False: ([p.id for p in selected], [p.text for p in selected]))
    retriever = upstream._ensure_retriever()
    dense = Mock(wraps=retriever.retrieve_passages_naive)
    monkeypatch.setattr(retriever, "retrieve_passages_naive", dense)
    generate = Mock(side_effect=lambda question, texts: " | ".join(texts))
    monkeypatch.setattr(upstream._answer_generator, "generate", generate)
    question = "What does Alice lead?"
    # Reverse only the passage ranking so we prove generation uses model order,
    # not the original graph/dense merge order.
    complete = upstream._reranker.client.chat.completions.create.side_effect
    def reverse_passages(**kwargs):
        response = complete(**kwargs)
        if "complete source passages" in kwargs["messages"][0]["content"]:
            ids = json.loads(response.choices[0].message.content)["ranked_ids"]
            response.choices[0].message.content = json.dumps({"ranked_ids": ids[::-1]})
        return response
    upstream._reranker.client.chat.completions.create.side_effect = reverse_passages
    retrieved = adapter.retrieve(question)
    expected = [p.text for p in retrieved.passages]
    generate.assert_not_called()
    assert retrieved.answer == ""
    dense.reset_mock()
    result = adapter.query(question)
    assert [p.text for p in result.passages] == expected
    assert len(expected) == len(set(expected)) == 3
    candidates = result.trace.events[-1]["candidate_passages"]
    assert expected == candidates[::-1][:3]
    generate.assert_called_once_with(question, expected)
    assert result.answer == " | ".join(expected)
    assert result.trace.events[0]["query_entities"] == ["Alice"]
    assert result.trace.events[0]["expanded_relations"]
    dense.assert_called_once_with(question, top_k=10, filter=None)
    assert result.trace.events[-1]["graph_passages"] == graph_count
    changed = EnglishUpstreamPipeline(upstream, adapter.store, "en",
                                      query_config=EnglishQueryConfig(relation_limit=32))
    assert changed.identity == adapter.identity  # Same index remains reusable.
    assert changed.fingerprint() != adapter.fingerprint()


@pytest.mark.parametrize("upstream", ["llm"], indirect=True)
def test_invalid_rerank_retains_filtered_evidence(tmp_path, upstream, monkeypatch):
    adapter = EnglishUpstreamPipeline(upstream, SnapshotStore(tmp_path / "catalog"), "en")
    passages = [Passage("p1", "allowed", "Alice leads Project A.", "en", "allowed", metadata={"title": "Allowed title"}),
                Passage("p2", "excluded", "Alice leads Project B.", "en", "excluded", metadata={"title": "Excluded title"})]
    adapter.build(passages, extract_triplets=False,
                  triplets={p.id: [["Alice", "leads", p.source]] for p in passages})
    response = SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content='{"ranked_ids": []}'))])
    upstream._reranker.client.chat.completions.create.side_effect = lambda **kw: response
    result = adapter.query("What does Alice lead?", filter='source == "allowed"')
    assert result.answer == passages[0].text
    assert [p.id for p in result.passages] == ["p1"]
    assert any(e.get("status") == "fallback" for e in result.trace.events)
    assert result.trace.events[-1]["dense_passages"] == 1
    ranking_input = json.loads(result.trace.events[-1]["passage_ranking_inputs"][0])
    assert ranking_input == {"sources": [{"source": "allowed", "title": "Allowed title"}], "text": passages[0].text}
    for call in upstream._reranker.client.chat.completions.create.call_args_list:
        assert "Project B" not in json.dumps(call.kwargs["messages"])
        assert "Excluded title" not in json.dumps(call.kwargs["messages"])
    empty = adapter.query("What does Alice lead?", filter='source == "absent"')
    assert empty.passages == []
    assert empty.status == "partial"
    assert empty.trace.events[-1]["dense_passages"] == 0


@pytest.mark.parametrize("upstream", ["llm"], indirect=True)
def test_nonjev_retrieval_error_still_propagates(tmp_path, upstream, monkeypatch):
    adapter = EnglishUpstreamPipeline(upstream, SnapshotStore(tmp_path / "catalog"), "en")
    adapter.build([Passage("p1", "one", "Alice leads Project A.", "en", "one")], extract_triplets=False,
                  triplets={"p1": [["Alice", "leads", "Project A"]]})
    monkeypatch.setattr(upstream._ensure_retriever(), "retrieve", Mock(side_effect=TimeoutError("fixture")))
    with pytest.raises(TimeoutError, match="fixture"):
        adapter.query("Who leads Project A?")
