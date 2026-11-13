import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
from helpers import FakeEmbedding, FakeModel, corpus
from vector_graph_rag_bilingualism import ChinesePipeline, SnapshotStore, ModelService, Trace
from vector_graph_rag_bilingualism.cli import main
from vector_graph_rag_bilingualism.providers import PROMPTS, Reranker


@pytest.fixture
def server():
    model, embedding = FakeModel(), FakeEmbedding()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/embeddings"):
                output = {"data": [{"index": i, "embedding": v} for i, v in enumerate(embedding.embed(body["input"]))]}
            elif self.path.endswith("/rerank"):
                output = {"results": [{"index": i} for i in reversed(range(len(body["documents"])))]}
            else:
                task = next(k for k, v in PROMPTS.items() if v == body["messages"][0]["content"])
                value = model.call(task, json.loads(body["messages"][1]["content"]), Trace())
                output = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
                          "usage": {"total_tokens": 10}}
            data = json.dumps(output).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_http_providers_and_cli_index_query(tmp_path, server, monkeypatch, capsys):
    monkeypatch.setenv("FIXTURE_KEY", "offline-test-value")
    service = {"base_url": server, "model": "fixture", "key_env": "FIXTURE_KEY"}
    config = {"language": "zh", "namespace": "cli-demo", "data_dir": str(tmp_path / "data"),
              "embedding": {"dimension": 16, "service": service}, "models": {"default": service}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    docs = tmp_path / "docs.jsonl"
    docs.write_text('\n'.join(json.dumps({"source": p.source, "text": p.text}) for p in corpus()))
    main(["--config", str(path), "index", "--documents", str(docs)])
    indexed = json.loads(capsys.readouterr().out)
    main(["--config", str(path), "query", "谁负责审批，期限多久？"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["snapshot_id"] == indexed["snapshot"]
    assert "offline-test-value" not in json.dumps(result)
    main(["--config", str(path), "verify-index"])
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert Reranker(ModelService(**service)).rank("q", ["a", "b"]) == [1, 0]


def test_http_api_routes_and_reports_tamper(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from vector_graph_rag_bilingualism.api import create_app
    pipeline = ChinesePipeline(SnapshotStore(tmp_path), "fixture", FakeEmbedding(), FakeModel())
    sid = pipeline.build(corpus())
    with TestClient(create_app({"fixture": {"zh": pipeline}})) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/query", json={"namespace": "fixture", "question": "审批期限？"}).json()["status"] == "complete"
        assert client.post("/v1/query", json={"namespace": "other", "question": "q"}).status_code == 400
        assert client.post("/v1/query", json={"namespace": "fixture", "question": "q", "language": "both"}).status_code == 400
        with pipeline.store.connection() as db:
            db.execute("DELETE FROM records WHERE snapshot=? AND id='a'", (sid,))
        assert client.post("/v1/query", json={"namespace": "fixture", "question": "审批？"}).status_code == 409
