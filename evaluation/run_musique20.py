"""Auditable 20-question MuSiQue smoke evaluation; never runs the full dataset.
Keeps pinned upstream retrieval/generation prompts. API wrapper only disables
thinking, caps output, validates completion and logs usage. No gold in retrieval.
"""
from pathlib import Path
import os, sys, json, random, hashlib, threading, time, argparse, shutil
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'.eval/musique20-20261113'
OUT.mkdir(parents=True,exist_ok=True)
LOCK=threading.Lock(); CTX=threading.local(); ACTIVE={'id':None}
CHAT='deepseek-ai/DeepSeek-V4.1-Flash'; JUDGE='deepseek-ai/DeepSeek-V4.1-Flash'
def sha(v): return hashlib.sha256(v).hexdigest()
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def save(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)); t.replace(p)
def append(name,v):
    with LOCK:
        with (OUT/name).open('a') as f: f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def restore_snapshot_identity(pipe, store, snapshot):
    manifest, _ = store.load(snapshot)
    recorded = manifest['config']
    relocated_settings = dict(recorded['settings'])
    if Path(relocated_settings['milvus_uri']).name == 'index.db':
        relocated_settings['milvus_uri'] = pipe.settings['milvus_uri']
    old_ner_cache = relocated_settings.get('ner_cache_dir')
    new_ner_cache = pipe.settings.get('ner_cache_dir')
    default_suffix = ('evaluation', 'data', 'ner_cache')
    if (old_ner_cache and new_ner_cache
            and Path(old_ner_cache).parts[-3:] == Path(new_ner_cache).parts[-3:] == default_suffix):
        relocated_settings['ner_cache_dir'] = new_ner_cache
    if {**recorded, 'settings': relocated_settings} != pipe.identity:
        raise ValueError('English upstream configuration changed; use a matching snapshot')
    pipe.identity = recorded

def load_keys():
    for line in (ROOT/'.env').read_text().splitlines():
        if '=' not in line or line.strip().startswith('#'):continue
        k,v=line.split('=',1);k=k.strip();v=v.strip().strip('"').strip("'")
        if k in ('VGRAG_CHAT_KEY','JEVHUB_API_KEY'):os.environ[k]=v
    if not all(os.environ.get(k) for k in ('VGRAG_CHAT_KEY','JEVHUB_API_KEY')):raise ValueError('Missing credentials')
def prepare(source):
    if (OUT/'manifest.json').exists():return
    allrows=rows(Path(source)); rng=random.Random(20261113); selected=[]
    for hops,n in [(2,8),(3,6),(4,6)]:
        pool=sorted([x for x in allrows if int(x['id'][0])==hops and x['answerable']],key=lambda x:x['id'])
        selected.extend(rng.sample(pool,n))
    docs={};questions=[]
    for q in selected:
        mapping={}
        for p in q['paragraphs']:
            text=p['title']+'\n\n'+p['paragraph_text']; pid='p_'+sha(text.encode())[:60]
            docs[pid]={'id':pid,'source':pid,'text':text,'language':'en','document_id':pid,'metadata':{'title':p['title']}}
            mapping[str(p['idx'])]=pid
        questions.append({**q,'hops':int(q['id'][0]),'paragraph_ids':mapping})
    for name,data in [('questions.jsonl',questions),('documents.jsonl',list(docs.values()))]:
        (OUT/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
    save(OUT/'manifest.json',{'seed':20261113,'dataset':'MuSiQue-Ans v1.0 dev','source':'https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/763b65f844118a148e92bb88e7de5cb191b4c5dc/musique_ans_v1.0_dev.jsonl','dataset_sha256':sha(Path(source).read_bytes()),'sample_sha256':sha((OUT/'questions.jsonl').read_bytes()),'documents_sha256':sha((OUT/'documents.jsonl').read_bytes()),'n':20,'hops':{'2':8,'3':6,'4':6},'documents':len(docs),'answer_model':CHAT,'judge_model':JUDGE,'protocol':'Shared union index; each question searches the complete shared index without a question-level passage-ID filter. No question decomposition, answer or supporting labels supplied to indexing/retrieval/answer. One passage per original paragraph; no additional splitting. Raw upstream answer scored with official normalization EM/F1; independent semantic judge reported separately.'})
    print('PREPARED',len(questions),len(docs),flush=True)
def attach(client,role):
    original=client.chat.completions.create
    def create(**kw):
        kw.update(extra_body={'enable_thinking':False},max_tokens=2048)
        started=time.monotonic(); ident=getattr(CTX,'id',None) or ACTIVE['id']
        try:
            kw.update(stream=True,stream_options={'include_usage':True})
            text=[];usage=None;finish=None
            with original(**kw) as stream:
                for chunk in stream:
                    if chunk.usage:usage=chunk.usage
                    for part in chunk.choices:
                        if part.delta.content:text.append(part.delta.content)
                        if part.finish_reason:finish=part.finish_reason
            response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=''.join(text)),finish_reason=finish)],usage=usage)
            choice=response.choices[0]
            append('llm.jsonl',{'id':ident,'role':role,'model':kw['model'],'messages':kw['messages'],'text':choice.message.content,'finish_reason':choice.finish_reason,'usage':response.usage.model_dump() if response.usage else None,'elapsed_s':round(time.monotonic()-started,3)})
            if choice.finish_reason!='stop' or not choice.message.content: raise ValueError('Incomplete model response')
            return response
        except Exception as e:
            append('errors.jsonl',{'id':ident,'role':role,'error_type':type(e).__name__,'http_status':getattr(e,'status_code',None)})
            raise
    client.chat.completions.create=create
    client.max_retries=0
    client.timeout=120.0
    return client

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.prepare)
    if not args.run:return
    load_keys()
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha((OUT/'questions.jsonl').read_bytes())==manifest['sample_sha256']
    assert sha((OUT/'documents.jsonl').read_bytes())==manifest['documents_sha256']
    model_home = OUT / 'vgrag-home'
    model_home.mkdir(exist_ok=True)
    cache = model_home / 'onnx-cache'
    if cache.is_symlink() and not cache.exists():
        cache.unlink()
    os.environ['VGRAG_HOME'] = str(model_home)
    from vector_graph_rag_bilingualism.english import EnglishUpstreamPipeline
    from vector_graph_rag_bilingualism.config import EnglishConfig
    from vector_graph_rag_bilingualism.storage import SnapshotStore
    from vector_graph_rag_bilingualism.models import Passage
    from openai import OpenAI
    os.environ['OPENAI_API_KEY']=os.environ['VGRAG_CHAT_KEY']
    model=CHAT
    settings={'milvus_uri':str(OUT/'index.db'),'collection_prefix':'musique20_v1','embedding_provider':'onnx','embedding_model':'qdrant/bge-large-en-v1.5-onnx','embedding_dimension':1024,'llm_model':model,'openai_base_url':'https://api.siliconflow.cn/v1','final_top_k':10,'use_llm_cache':False,'batch_size':16,'milvus_consistency_level':'Bounded','milvus_index_type':'AUTOINDEX'}
    store=SnapshotStore(OUT/'snapshots')
    pipe=EnglishUpstreamPipeline.connect(store,'musique20',EnglishConfig(settings=settings),reranker={'provider':'jev','service':'jevhub','model':'jev-1.13.0','max_concurrency':2})
    save(OUT/'settings.json',pipe.settings)
    pipe.jev_accounting.observer=lambda event:append('jev-http.jsonl',{'id':ACTIVE['id'],**event})
    upstream=pipe.upstream
    upstream._triplet_extractor.cache=None;upstream._triplet_extractor.use_cache=False
    upstream._answer_generator.cache=None;upstream._answer_generator.use_cache=False
    attach(upstream._triplet_extractor.client,'triplets');attach(upstream._answer_generator.client,'answer')
    if not (OUT/'health.json').exists():
        health=attach(OpenAI(api_key=os.environ['VGRAG_CHAT_KEY'],base_url='https://api.siliconflow.cn/v1',max_retries=0),'health')
        health.chat.completions.create(model=CHAT,messages=[{'role':'user','content':'Reply with OK.'}],temperature=0)
        save(OUT/'health.json',{'ok':True});health.close()
    passages=[Passage(**p) for p in rows(OUT/'documents.jsonl')]
    if not (OUT/'snapshot.json').exists():
        cached={x['id']:x for x in rows(OUT/'triplets.jsonl')}
        pending=[p for p in passages if p.id not in cached]
        def extract(p):
            CTX.id=p.id
            ts=upstream._triplet_extractor.extract(p.text)
            v={'id':p.id,'text_sha256':sha(p.text.encode()),'triplets':[[t.subject,t.predicate,t.object] for t in ts],'empty':not ts}
            append('triplets.jsonl',v);return v
        with ThreadPoolExecutor(max_workers=6) as ex:
            for fut in as_completed([ex.submit(extract,p) for p in pending]):
                v=fut.result();cached[v['id']]=v
                if len(cached)%20==0:print('TRIPLETS',len(cached),'/',len(passages),flush=True)
        for p in passages:assert cached[p.id]['text_sha256']==sha(p.text.encode())
        print('INDEXING',len(passages),flush=True)
        snapshot=pipe.build(passages,extract_triplets=False,triplets={k:v['triplets'] for k,v in cached.items()})
        save(OUT/'snapshot.json',{'snapshot':snapshot})
    pipe.snapshot=json.loads((OUT/'snapshot.json').read_text())['snapshot']
    restore_snapshot_identity(pipe, store, pipe.snapshot)
    ner=upstream._ensure_retriever().entity_extractor
    ner.cache=None;ner.use_cache=False
    attach(ner.client,'entities')
    done={x['id']:x for x in rows(OUT/'answers.jsonl')}
    questions=rows(OUT/'questions.jsonl');assert len(questions)==20
    for q in questions:
        if q['id'] in done:continue
        ACTIVE['id']=q['id'];started=time.monotonic()
        result=pipe.query(q['question'])
        selected={p.id for p in result.passages}
        gold={q['paragraph_ids'][str(p['idx'])] for p in q['paragraphs'] if p['is_supporting']}
        record={'id':q['id'],'hops':q['hops'],'result':result.as_dict(),'gold_answer':q['answer'],'answer_aliases':q['answer_aliases'],'selected_ids':sorted(selected),'gold_ids':sorted(gold),'support_recall':len(selected&gold)/len(gold),'all_support':gold<=selected,'elapsed_s':round(time.monotonic()-started,2)}
        append('answers.jsonl',record);done[q['id']]=record
        print('ANSWER',len(done),'/20',q['id'],'support',record['support_recall'],'answer',result.answer[:180].replace('\n',' '),flush=True)
    judge=attach(OpenAI(api_key=os.environ['VGRAG_CHAT_KEY'],base_url='https://api.siliconflow.cn/v1',max_retries=0),'judge')
    judged={x['id']:x for x in rows(OUT/'scores.jsonl')}
    for q in questions:
        if q['id'] in judged:continue
        ACTIVE['id']=q['id'];a=done[q['id']]
        instruction='Evaluate an English multi-hop QA answer. Return JSON with correct (boolean), reason (string), ambiguity (boolean). Judge factual equivalence to the reference and aliases in context of the question. Full sentences, punctuation, bold text, abbreviations and unambiguous equivalent names are acceptable; a missing/wrong requested fact or wrong entity is incorrect. Do not reward mere keyword overlap or repair an incorrect answer. Flag ambiguity when the reference conflicts with the provided supporting paragraphs or a defensible alternative exists. Do not change the reference score because of ambiguity.'
        payload={'question':q['question'],'reference':q['answer'],'aliases':q['answer_aliases'],'prediction':a['result']['answer'],'reference_support':[p for p in q['paragraphs'] if p['is_supporting']]}
        response=judge.chat.completions.create(model=JUDGE,messages=[{'role':'system','content':instruction},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],response_format={'type':'json_object'},temperature=0)
        score=json.loads(response.choices[0].message.content);assert type(score.get('correct')) is bool
        append('scores.jsonl',{'id':q['id'],**score});judged[q['id']]=score
        print('SCORE',len(judged),'/20',q['id'],score['correct'],flush=True)
    save(OUT/'completion.json',{'answers':len(done),'scores':len(judged),'completed':True})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
