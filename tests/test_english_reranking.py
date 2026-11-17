import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vector_graph_rag_bilingualism.config import EnglishQueryConfig
from vector_graph_rag_bilingualism.english_reranking import GenerativeRanker, interleave


def response(content, finish="stop"):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content))])


def ranker(outputs, config=None):
    client = Mock(base_url="https://api.siliconflow.cn/v1")
    client.with_options.return_value = client
    client.chat.completions.create.side_effect = outputs
    return GenerativeRanker(client, "deepseek-ai/DeepSeek-V4.1-Flash", config or EnglishQueryConfig()), client


@pytest.mark.parametrize("bad", [
    '{"ranked_ids": [', '{}', '[]', '{"ranked_ids": []}',
    '{"ranked_ids": ["c0", "c0"]}', '{"ranked_ids": ["c0", "invented"]}',
    '{"ranked_ids": [0, 1]}', '{"ranked_ids": "c0,c1"}',
])
def test_invalid_schema_retries_then_preserves_candidates(bad):
    model, client = ranker([response(bad), response(bad)])
    order, events = model.rank("question", ["first", "second"], stage="passage_rerank", limit=2)
    assert order == [0, 1]
    assert [e["status"] for e in events] == ["retry", "fallback", "selected"]
    assert events[0]["raw_response"] == bad
    assert client.chat.completions.create.call_count == 2


def test_length_stop_is_retried_even_for_parseable_json():
    body = '{"ranked_ids": ["c1", "c0"]}'
    model, client = ranker([response(body, "length"), response(body)])
    order, events = model.rank("question", ["one", "two"], stage="relation_rerank", limit=2)
    assert order == [1, 0]
    assert events[0]["finish_reason"] == "length"
    assert events[0]["status"] == "retry"
    assert events[1]["status"] == "complete"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "stop" not in kwargs
    assert kwargs["extra_body"] == {"enable_thinking": False}
    assert kwargs["max_tokens"] == 4096
    client.with_options.assert_called_once_with(max_retries=0, timeout=120)


def test_relations_keep_wide_candidate_set_instead_of_five():
    ids = [f"c{i}" for i in range(69, 5, -1)]
    model, _ = ranker([response(json.dumps({"ranked_ids": ids}))])
    order, events = model.rank("multi hop", [f"relation {i}" for i in range(70)],
                               stage="relation_rerank", limit=64)
    assert order == list(range(69, 5, -1))
    assert events[0]["status"] == "complete"


def test_batches_keep_whole_text_and_merge_without_duplicates():
    received = []
    def complete(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        received.extend(x["text"] for x in payload["candidates"])
        ids = [x["id"] for x in payload["candidates"]][::-1][:payload["count"]]
        return response(json.dumps({"ranked_ids": ids}))
    texts = [str(i) + "a" * 800 for i in range(6)]
    model, _ = ranker(complete, EnglishQueryConfig(batch_chars=2300))
    order, events = model.rank("q", texts, stage="passage_rerank", limit=6)
    assert received == texts
    assert sorted(order) == list(range(6))
    assert events[-1]["batches"] > 1
    assert events[-1]["merge"] == "round_robin"


def test_oversized_passage_and_empty_corpus_make_no_requests():
    model, client = ranker([], EnglishQueryConfig(batch_chars=1000))
    assert model.rank("q", [], stage="passage_rerank", limit=16) == ([], [])
    order, events = model.rank("q", ["x" * 2000], stage="passage_rerank", limit=1)
    assert order == [0]
    assert events[0]["error"] == "candidate_exceeds_batch_chars"
    client.chat.completions.create.assert_not_called()


def test_transport_failure_does_not_leak_exception_body():
    model, _ = ranker([RuntimeError("fixture-secret"), RuntimeError("fixture-secret")])
    order, events = model.rank("q", ["a"], stage="passage_rerank", limit=1)
    assert order == [0]
    assert "fixture-secret" not in json.dumps(events)


def test_graph_dense_merge_keeps_both_channels_and_caps_union():
    assert interleave(["a", "b", "c"], ["b", "d", "e"], 4) == ["a", "b", "d", "c"]


@pytest.mark.parametrize("config", [{"relation_limit": 0}, {"dense_top_k": True}, {"batch_chars": 1.5}])
def test_invalid_query_limits(config):
    with pytest.raises(ValueError):
        EnglishQueryConfig(**config)


def test_cli_passes_query_configuration(tmp_path, monkeypatch):
    from vector_graph_rag_bilingualism import cli
    connect = Mock(return_value="pipeline")
    monkeypatch.setattr(cli.EnglishUpstreamPipeline, "connect", connect)
    result = cli.connect({"language": "en", "namespace": "demo", "data_dir": str(tmp_path),
                          "upstream": {}, "query": {"relation_limit": 32}})
    assert result == "pipeline"
    assert connect.call_args.args[2].query.relation_limit == 32


def test_responses_api_also_checks_completion():
    model, client = ranker([])
    model.model = "gpt-6-fixture"
    body = '{"ranked_ids": ["c0"]}'
    client.responses.create.side_effect = [SimpleNamespace(status=status, output_text=body, usage=None)
                                           for status in ("incomplete", "completed")]
    order, events = model.rank("q", ["a"], stage="passage_rerank", limit=1)
    assert order == [0]
    assert events[0]["status"] == "retry"
    assert events[0]["reason"] == "incomplete_response"
    assert events[1]["status"] == "complete"
