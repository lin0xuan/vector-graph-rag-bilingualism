import json
from dataclasses import replace
import pytest
from vector_graph_rag_bilingualism import ChinesePipeline, ChineseConfig, SnapshotStore, IntegrityError, Router
from vector_graph_rag_bilingualism.ingestion import ingest_text
from helpers import FakeEmbedding, FakeModel, corpus


@pytest.fixture
def pipeline(tmp_path):
    model = FakeModel()
    pipeline = ChinesePipeline(SnapshotStore(tmp_path), "zh-fixture", FakeEmbedding(), model)
    pipeline.build(corpus())
    return pipeline


def test_complete_with_located_evidence_and_original_query_channel(pipeline):
    result = pipeline.query("谁负责审批，期限多久？")
    assert result.status == "complete"
    assert not result.missing_slots
    # Sentence selection marks evidence; remaining ranked paragraphs may still
    # enter the answer context when the budget permits.
    assert [p.id for p in result.passages] == ["a", "b", "c"]
    assert {e.passage_id for e in result.evidence} == {"a", "b"}
    assert all(e.quote == next(p.text[e.start:e.end] for p in result.passages if p.id == e.passage_id) for e in result.evidence)
    assert any(e.get("slot") == "original" for e in result.trace.events)


def test_answer_omission_is_repaired_once(pipeline):
    pipeline.model.omit_first_answer = True
    result = pipeline.query("谁负责审批，期限多久？")
    assert result.status == "complete"
    assert "期限" in result.answer
    assert pipeline.model.calls["answer"] == 2
    assert pipeline.model.calls["verify"] == 2
    assert any(e["stage"] == "answer_repair" for e in result.trace.events)


def test_entity_failure_is_visible_and_does_not_erase_evidence(pipeline):
    pipeline.model.fail_entities = True
    result = pipeline.query("谁负责审批，期限多久？")
    assert result.status == "partial"
    assert [p.id for p in result.passages] == ["a", "b", "c"]
    assert {e.passage_id for e in result.evidence} == {"a", "b"}
    assert any(e["stage"] == "entities" and e["status"] == "failed" for e in result.trace.events)


def test_unlocated_quotes_cannot_count_as_coverage(pipeline):
    pipeline.model.invalid_quote = True
    result = pipeline.retrieve("谁负责审批，期限多久？")
    assert result.missing_slots == ["s1", "s2"]
    assert not result.evidence
    assert result.status == "partial"
    assert result.passages  # Invalid selection keeps unconfirmed originals.
    assert sum(e["stage"] == "repair" for e in result.trace.events) <= 2


@pytest.mark.parametrize("field", ["text", "metadata", "vector", "relations"])
def test_same_count_mutation_blocked_before_model_calls(pipeline, field):
    snapshot = pipeline.store.current(pipeline.namespace)
    _, records = pipeline.store.load(snapshot)
    record = records[0]
    if field in ("text", "metadata"):
        record["passage"][field] = "改写" if field == "text" else {"tampered": True}
    elif field == "vector":
        record["vector"][0] += 0.1
    else:
        record["relations"].append({"subject": "伪造"})
    with pipeline.store.connection() as db:
        db.execute("UPDATE records SET payload=? WHERE snapshot=? AND id=?", (json.dumps(record), snapshot, record["passage"]["id"]))
    before = dict(pipeline.model.calls)
    with pytest.raises(IntegrityError):
        pipeline.query("审批期限？")
    assert dict(pipeline.model.calls) == before


def test_two_query_profiles_reuse_identical_index(pipeline):
    snapshot = pipeline.store.current(pipeline.namespace)
    first = pipeline.query("谁负责审批，期限多久？")
    second = ChinesePipeline(pipeline.store, pipeline.namespace, pipeline.embedder, FakeModel(),
                             config=ChineseConfig(max_repairs=0, profile="zh_no_repair"), snapshot=snapshot)
    other = second.query("谁负责审批，期限多久？")
    assert first.snapshot_id == other.snapshot_id == snapshot
    assert first.answer == other.answer
    assert pipeline.fingerprint() != second.fingerprint()


def test_embedding_mismatch_blocks_query(pipeline):
    pipeline.embedder.identity = lambda: {"model": "different"}
    before = dict(pipeline.model.calls)
    with pytest.raises(ValueError, match="Embedding"):
        pipeline.query("审批？")
    assert dict(pipeline.model.calls) == before


def test_failed_extraction_keeps_original_and_telemetry(tmp_path):
    model = FakeModel()
    model.fail_extract = True
    pipeline = ChinesePipeline(SnapshotStore(tmp_path), "zh", FakeEmbedding(), model)
    sid = pipeline.build(corpus())
    manifest, rows = pipeline.store.load(sid)
    assert len(rows) == 3
    assert sum(e.get("status") == "failed" for e in manifest["telemetry"]) == 3
    assert pipeline.query("谁负责审批，期限多久？").passages


def test_snapshot_activation_namespace_and_old_snapshot(pipeline):
    old = pipeline.store.current(pipeline.namespace)
    new = pipeline.build(corpus()[:2])
    assert new != old
    assert len(pipeline.store.load(old)[1]) == 3
    assert len(pipeline.store.load(new)[1]) == 2
    with pytest.raises(ValueError):
        pipeline.store.activate("other", old)


def test_routing_requires_explicit_language_for_mixed_collection(pipeline):
    assert Router({"zh": pipeline}).query("审批期限？").language == "zh"
    router = Router({"zh": pipeline, "en": object()})
    with pytest.raises(ValueError, match="Mixed"):
        router.query("审批期限？")
    with pytest.raises(NotImplementedError):
        router.query("question", language="both")


def test_ingestion_links_only_same_article_and_preserves_offsets():
    chunks = ingest_text("source", "这是审批规则。办理期限为三个工作日。第二段保留资料。", target=10, maximum=14)
    assert len(chunks) >= 2
    ids = {p.id for p in chunks}
    assert all(set(p.neighbors) <= ids for p in chunks)
    assert all(p.metadata["end_offset"] > p.metadata["start_offset"] for p in chunks)
    other = ingest_text("source", "这是另一篇文章。", document_id="separate")
    assert not other[0].neighbors
    assert other[0].document_id != chunks[0].document_id


def test_neighbor_repair_is_targeted_and_stays_in_article(pipeline, monkeypatch):
    from vector_graph_rag_bilingualism import Passage
    from vector_graph_rag_bilingualism.retrieval import Retriever
    p1 = Passage("a", "policy", "张某负责审批。", "zh", "article", neighbors=["b", "foreign"])
    p2 = Passage("b", "policy", "办理期限为三个工作日。", "zh", "article")
    p3 = Passage("foreign", "policy", "别的事件办理期限为七日。", "zh", "another_article")
    pipeline.build([p1, p2, p3])
    questions = []
    def only_first(self, queries, entities, config, trace):
        questions.append(queries)
        return ["a"]
    monkeypatch.setattr(Retriever, "collect", only_first)
    result = pipeline.query("谁负责审批，期限多久？")
    assert result.status == "complete"
    repairs = [e for e in result.trace.events if e["stage"] == "repair"]
    assert repairs[0]["added"] == ["b"]
    assert [slot for slot, _ in questions[1]] == ["s2"]
    assert "foreign" not in [p.id for p in result.passages]


def test_index_resume_skips_completed_extraction_and_can_retry_failed(pipeline):
    before = pipeline.model.calls["extract"]
    pipeline.build(corpus())
    assert pipeline.model.calls["extract"] == before
    pipeline.model.fail_extract = True
    pipeline.build(corpus(), resume=False)
    before = pipeline.model.calls["extract"]
    pipeline.model.fail_extract = False
    sid = pipeline.build(corpus(), retry_failed=True)
    assert pipeline.model.calls["extract"] == before + 3
    manifest, _ = pipeline.store.load(sid)
    assert all(e.get("status") != "failed" for e in manifest["telemetry"])


def test_empty_and_failing_selection_do_not_generate_complete_answer(pipeline):
    pipeline.model.fail_select = True
    result = pipeline.query("审批期限？")
    assert result.status == "partial"
    assert result.passages  # Invalid selection keeps unconfirmed originals.
    assert any(e.get("stage") == "selection" and e.get("status") == "failed" for e in result.trace.events)


def test_numeric_copy_error_is_repaired_even_if_semantic_verifier_accepts(pipeline):
    from vector_graph_rag_bilingualism.models import Trace
    pipeline.model.call = lambda task, data, trace: {"ok": True, "issues": []}
    issues = pipeline._verify("总额138134元。[p1]", {"passages": [{"id": "p1", "text": "总额1,138,134元。"}]}, Trace())
    assert any("numeric_copy_mismatch" in issue for issue in issues)


def test_relation_extraction_vectors_and_two_stage_retrieval(pipeline):
    original = pipeline.model.call
    def model(task, data, trace):
        if task == "extract" and "张某" in data["text"]:
            return {"relations": [{"subject": "张某", "predicate": "负责", "object": "审批", "quote": "张某负责审批。"}]}
        return original(task, data, trace)
    pipeline.model.call = model
    sid = pipeline.build(corpus(), resume=False)
    _, records = pipeline.store.load(sid)
    assert any(r["relations"] and r["relations"][0]["embedding"] for r in records)
    result = pipeline.query("谁负责审批，期限多久？")
    assert result.status == "complete"
    assert pipeline.model.calls["relations"] > 0
    assert any(e["stage"] == "relation_rerank" for e in result.trace.events)


def test_single_and_two_stage_share_graph_snapshot(pipeline):
    original = pipeline.model.call
    def model(task, data, trace):
        if task == "extract" and "张某" in data["text"]:
            return {"relations": [{"subject": "张某", "predicate": "负责", "object": "审批", "quote": data["text"]}]}
        return original(task, data, trace)
    pipeline.model.call = model
    snapshot = pipeline.build(corpus(), resume=False)
    two = pipeline.query("谁负责审批，期限多久？")
    assert pipeline.model.calls["relations"] > 0
    single_model = FakeModel()
    single = ChinesePipeline(pipeline.store, pipeline.namespace, pipeline.embedder, single_model,
                             config=ChineseConfig(retrieval_mode="single-stage"), snapshot=snapshot)
    one = single.query("谁负责审批，期限多久？")
    assert single_model.calls["relations"] == 0
    assert one.snapshot_id == two.snapshot_id
    assert one.answer == two.answer
    assert any(e.get("effective_mode") == "two-stage" for e in two.trace.events)
    assert any(e.get("effective_mode") == "single-stage" for e in one.trace.events)
