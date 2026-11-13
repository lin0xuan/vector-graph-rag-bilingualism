import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
from helpers import FakeEmbedding, FakeModel, corpus
from vector_graph_rag_bilingualism import ChinesePipeline, SnapshotStore
from vector_graph_rag_bilingualism.jev import JevReranker


def test_missing_key_fails_before_any_service(monkeypatch):
    monkeypatch.delenv('MISSING_JEV_KEY',raising=False)
    with pytest.raises(ValueError,match='credential'):
        JevReranker(key_env='MISSING_JEV_KEY')


def test_real_native_protocol_for_both_chinese_stages(tmp_path,monkeypatch):
    pytest.importorskip('vector_graph_rag')
    import tiktoken
    class Tokens:
        def encode(self,text,**kwargs): return list(text)
    monkeypatch.setattr(tiktoken,'get_encoding',lambda name:Tokens())
    monkeypatch.setenv('FIXTURE_JEV_KEY','offline-key')
    seen=[]; broken=[False]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            assert self.path=='/v1/systemone'
            assert self.headers['Authorization']=='Bearer offline-key'
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            output={'answers':{k:{'type':'noul','noul':0.7} for k in body['questions']}}
            if broken[0]: output['answers']={}
            data=json.dumps(output).encode(); self.send_response(200)
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        adapter=JevReranker(key_env='FIXTURE_JEV_KEY',base_url=f'http://127.0.0.1:{server.server_port}/v1')
        model=FakeModel(); original=model.call
        def call(task,data,trace):
            if task=='extract' and '张某' in data['text']:
                return {'relations':[{'subject':'张某','predicate':'负责','object':'审批','quote':data['text']}]}
            return original(task,data,trace)
        model.call=call
        p=ChinesePipeline(SnapshotStore(tmp_path),'jev-test',FakeEmbedding(),model,reranker=adapter,relation_reranker=adapter)
        sid=p.build(corpus()); result=p.query('谁负责审批，期限多久？')
        assert result.snapshot_id==sid and result.status=='complete'
        assert model.calls['relations']==0
        assert any('candidate_graph' in b['state'] for b in seen)
        assert any('excerpts' in b['state'] for b in seen)
        # More than 16 candidates and identical texts retain every identity.
        assert adapter.rank('审批',['相同原文']*20)==list(range(20))
        assert all(len(b['state'].get('excerpts',{}))<=16 for b in seen)
        assert 'offline-key' not in json.dumps(adapter.identity())
        broken[0]=True
        with pytest.raises(ValueError,match='missing'):
            adapter.rank('审批',['原文'])
        failed=p.query('谁负责审批，期限多久？')
        assert failed.status=='partial' and failed.passages
        assert any(e.get('stage')=='passage_rerank' and e.get('status')=='fallback' for e in failed.trace.events)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
