import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
import pytest
from vector_graph_rag_bilingualism.jev import JevReranker
from vector_graph_rag_bilingualism.jev_service import resolve_service, model_connection, JevAccounting
from vector_graph_rag_bilingualism.providers import SelectiveReranker


def test_selective_reranker_limits_remote_calls_per_query():
    class Delegate:
        def __init__(self): self.calls = []
        def identity(self): return {"provider": "fixture"}
        def rank(self, question, texts): self.calls.append(("passages", question)); return list(reversed(range(len(texts))))
        def rank_relations(self, question, texts): self.calls.append(("relations", question)); return list(reversed(range(len(texts))))

    delegate = Delegate()
    reranker = SelectiveReranker(delegate, min_candidates=3, max_calls_per_query=1)
    assert reranker.rank_relations("q", ["a", "b"]) == [0, 1]
    assert reranker.rank("q", ["a", "b", "c"]) == [2, 1, 0]
    assert reranker.rank("q", ["a", "b", "c", "d"]) == [0, 1, 2, 3]
    assert delegate.calls == [("passages", "q")]
    assert reranker.rank("q2", ["a", "b", "c"]) == [2, 1, 0]


def test_defaults_and_no_cross_service_credentials(monkeypatch):
    official = resolve_service()
    gateway = resolve_service(service="jevhub")
    assert official["base_url"] == "https://api.typesafe.ai/v1"
    assert official["key_env"] == "TYPESAFE_API_KEY"
    assert gateway["base_url"] == "https://api.jevgo.cn/v1"
    assert gateway["key_env"] == "JEVHUB_API_KEY"
    monkeypatch.setenv("TYPESAFE_API_KEY", "official-only")
    monkeypatch.delenv("JEVHUB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="JEVHUB_API_KEY"):
        model_connection(gateway)
    monkeypatch.setenv("JEVHUB_API_KEY", "gateway-only")
    assert model_connection(official)["api_key"] == "official-only"
    assert model_connection(gateway)["api_key"] == "gateway-only"
    with pytest.raises(ValueError, match="Unknown"):
        resolve_service(service="guess-from-key")
    with pytest.raises(ValueError, match="Pin"):
        resolve_service(service="jevhub", model="jev-latest")
    with pytest.raises(ValueError, match="at most 4"):
        resolve_service(service="jevhub", max_concurrency=5)


@pytest.mark.parametrize("service", ["typesafe", "jevhub"])
def test_both_protocols_native_english_and_chinese_and_accounting(monkeypatch, service):
    pytest.importorskip("vector_graph_rag")
    import tiktoken
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: SimpleNamespace(encode=lambda text, **kw: list(text)))
    monkeypatch.setenv("FIXTURE_KEY", "fixture-secret")
    seen, events = [], []
    state = {"mode": "good"}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            assert self.path == "/v1/systemone"
            assert self.headers["Authorization"] == "Bearer fixture-secret"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            code = 401 if state["mode"] == "unauthorized" else 200
            data = {"model": body["model"], "answers": {k: {"type": "noul", "noul": .8} for k in body["questions"]}}
            if state["mode"] == "bad_scores": data["answers"] = {}
            if service == "typesafe": data["usage"] = {"input_tokens": 100, "output_tokens": 4}
            raw = json.dumps(data).encode()
            self.send_response(code)
            if service == "jevhub" and code == 200:
                self.send_header("X-JevHub-Charge-USD", "0.000123")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers(); self.wfile.write(raw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        adapter = JevReranker(service=service, base_url=f"http://127.0.0.1:{server.server_port}/v1/",
                              key_env="FIXTURE_KEY", request_observer=events.append)
        assert adapter.rank_relations("审批", ["张某负责审批", "其他背景"]) == [0, 1]
        assert adapter.rank("审批", ["原文"]*20) == list(range(20))
        assert adapter.native.rerank("approval", ["1", "2"], ["Alice approves", "background"])[0] == ["1", "2"]
        assert adapter.native.rerank_passages("approval", ["Alice approves", "background"]) == ["Alice approves", "background"]
        assert len(events) == 5
        assert {e["stage"] for e in events} == {"relations", "passages"}
        if service == "jevhub":
            assert all(e["usage"] is None for e in events)
            assert adapter.accounting.snapshot()["reported_charge_usd"] == "0.000615"
        else:
            assert all(e["charge_usd"] is None for e in events)
            assert adapter.accounting.snapshot()["reported_usage"] == {"input_tokens":500, "output_tokens":20}
        assert "fixture-secret" not in json.dumps(events)+json.dumps(adapter.identity())
        # Paid HTTP 200 with invalid scores is still counted, then rejected.
        state["mode"] = "bad_scores"
        with pytest.raises(ValueError): adapter.rank("审批", ["原文"])
        assert len(events) == 6
        state["mode"] = "unauthorized"
        with pytest.raises(Exception): adapter.rank("审批", ["原文"])
        assert len(events) == 7 and events[-1]["http_status"] == 401
        assert events[-1]["charge_usd"] is None
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


@pytest.mark.parametrize("service", ["typesafe", "jevhub"])
def test_english_connect_passes_explicit_service_without_leaking_key(monkeypatch, service):
    upstream = pytest.importorskip("vector_graph_rag")
    from vector_graph_rag_bilingualism.english import EnglishUpstreamPipeline, safe_settings
    from vector_graph_rag_bilingualism.config import EnglishConfig
    captured = {}
    def construct(*, settings):
        captured["settings"] = settings
        return SimpleNamespace(_reranker=SimpleNamespace(_request=lambda *a: None))
    monkeypatch.setattr(upstream, "VectorGraphRAG", construct)
    monkeypatch.setattr(EnglishUpstreamPipeline, "__init__", lambda *a, **k: None)
    options = resolve_service(service=service)
    monkeypatch.setenv(options["key_env"], "fixture-secret")
    config = EnglishConfig(dict(embedding_provider="openai", embedding_model="fixture", embedding_dimension=8,
                                llm_model="fixture", final_top_k=3, collection_prefix="fixture"))
    result = EnglishUpstreamPipeline.connect(None, "fixture", config, reranker={"provider":"jev", "service":service})
    connection = captured["settings"].task_model("reranker")
    assert connection.base_url == options["base_url"]
    assert connection.api_key.get_secret_value() == "fixture-secret"
    assert "fixture-secret" not in json.dumps(safe_settings(captured["settings"]))
    assert isinstance(result.jev_accounting, JevAccounting)
