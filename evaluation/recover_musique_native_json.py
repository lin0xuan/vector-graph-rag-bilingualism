"""Offline replay of native invalid-JSON semantics, only after exhausted retries.
Does not invent triplets, discard documents, call a model or inspect gold labels.
Only use after the evaluation worker has exited.
"""
from pathlib import Path
import sys,json,hashlib,collections,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vector_graph_rag_bilingualism.english import verify_upstream_version
from vector_graph_rag.llm.extractor import TripletExtractor
verify_upstream_version()
P=ROOT/'.eval/musique20-20261113'
def rows(name):
 p=P/name;return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
done={x['id'] for x in rows('triplets.jsonl')};errors=collections.Counter(x['id'] for x in rows('errors.jsonl') if x['role']=='triplets')
responses=collections.defaultdict(list)
for r in rows('llm.jsonl'):
 if r['role']=='triplets':responses[r['id']].append(r)
for d in rows('documents.jsonl'):
 if d['id'] in done or errors[d['id']]<3:continue
 candidates=[r for r in responses[d['id']] if r['finish_reason']=='length' and r['text']]
 if not candidates:continue
 r=candidates[-1];ts=TripletExtractor._parse_response(None,r['text'])
 assert not ts, 'Only replay malformed JSON as the native empty-triplet result'
 record={'id':d['id'],'text_sha256':hashlib.sha256(d['text'].encode()).hexdigest(),'triplets':[],'empty':True,'extraction_failed':True,'recovery':'Pinned upstream _parse_response replay on malformed length-limited response after >=3 failed attempts; original passage retained','failed_attempts':errors[d['id']],'response_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),'parser_sha256':hashlib.sha256(inspect.getsource(TripletExtractor._parse_response).encode()).hexdigest()}
 with (P/'triplets.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
 print('NATIVE_EMPTY_RELATIONS',d['id'],d['metadata']['title'])
