"""Resumable 1,000-question Chinese no-Jev evaluation.

Reuses exactly the historical light 50 double + 50 triple questions. Runtime
pipeline and model budgets match the Jev 20-question run; only sample size changes.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
import compare_light as base
from retest_jev import JUDGE_PROMPT


def load_chat_key(path):
    """Read this project's VGRAG_CHAT_KEY without shell interpolation or key logging."""
    value = None
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, separator, raw = line.partition('=')
        if separator and name.strip() == 'VGRAG_CHAT_KEY':
            value = raw.strip().strip('"\'')
    if not value:
        raise ValueError('VGRAG_CHAT_KEY is missing or empty in the model env file')
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--historical',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--env-file',type=Path,required=True)
    ap.add_argument('--model-env-file',type=Path,default=Path('.env'))
    ap.add_argument('--embedding-cache',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    if not 1<=args.workers<=4 or args.output.resolve() in (args.previous.resolve(),args.historical.resolve()):
        ap.error('Use a separate output and 1–4 workers')
    args.output.mkdir(parents=True,exist_ok=True)
    from vector_graph_rag_bilingualism.locks import exclusive_lock
    with exclusive_lock(args.output/'run.lock'):
        execute(args)


def execute(args):
    from vector_graph_rag_bilingualism.cli import load_env
    from fastembed import TextEmbedding
    from tokenizers import Tokenizer
    from vector_graph_rag_bilingualism import ChinesePipeline, ChineseConfig, SnapshotStore
    from vector_graph_rag_bilingualism.config import PIPELINE_VERSION
    from vector_graph_rag_bilingualism.metrics import diagnostics
    from vector_graph_rag_bilingualism.providers import PROMPTS
    load_env(args.env_file)
    root=Path(__file__).resolve().parents[1]
    before=json.loads((args.previous/'manifest.json').read_text())
    assert json.loads((args.previous/'integrity-after.json').read_text())['verified']
    previous_files=['manifest.json','questions.jsonl','generations.jsonl','grades.jsonl','integrity-after.json']
    previous_hashes={name:base.sha(args.previous/name) for name in previous_files}
    historical_files=[]
    historical_hashes={}
    sample=base.rows(args.historical/'questions.jsonl')
    assert len(sample)==len({q['id'] for q in sample})==1000
    assert Counter(len(q['sources']) for q in sample)=={2:500,3:500}
    assert base.sha(root/'evaluation/compare_light.py')==before['adapter_sha256']
    assert hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()==before['judge_prompt_sha256']
    index=args.output/'new-index';index.mkdir(exist_ok=True)
    if not (index/'catalog.sqlite3').exists():
        shutil.copy2(args.previous/'new-index/catalog.sqlite3',index/'catalog.sqlite3')
    store=SnapshotStore(index);snapshot=before['new_snapshot'];stored,records=store.load(snapshot)
    assert stored['content_hash']==before['index_content_hash']
    assert len(records)==2993 and len({r['passage']['source'] for r in records})==2309
    assert all(set(q['sources'])<={r['passage']['source'] for r in records} for q in sample)
    os.environ.pop('HF_HUB_OFFLINE', None);os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    embedding=TextEmbedding('BAAI/bge-small-zh-v1.5',cache_dir=str(args.embedding_cache),threads=2)
    tokenizer=Tokenizer.from_str(embedding.model.tokenizer.to_str());tokenizer.no_truncation();tokenizer.no_padding()
    embedder=base.SharedEmbedding(SimpleNamespace(model=embedding,embed_lock=threading.Lock()))
    assert stored['config']['embedding']==embedder.identity()
    class GuardedClient(base.RealRequests):
        def ask(self,*a,**kw):
            try: return super().ask(*a,**kw)
            except HTTPError as error:
                raise
    client=GuardedClient(load_chat_key(args.model_env_file),args.output,tokenizer)
    model=base.RealJsonModel(client,PROMPTS)
    assert model.identity()==before['model']
    config=ChineseConfig();assert asdict(config)==before['new_config']
    manifest={'schema':1,'experiment':'no-jev-fixed-model-1000','questions':1000,
        'previous':str(args.previous.resolve()),'previous_files':previous_hashes,
        'historical':None,'historical_files':historical_hashes,
        'sample_sha256':base.sha(args.historical/'questions.jsonl'),'sample_selection':'Exact official CRUD-RAG full sample; 500 double + 500 triple',
        'new_snapshot':snapshot,'index_content_hash':stored['content_hash'],'documents':2309,'passages':2993,
        'query_model':base.CHAT,'judge_model':base.JUDGE,'temperature':0,'enable_thinking':False,
        'answer_max_tokens':2000,'jev_enabled':False,'jev':None,'new_config':asdict(config),
        'pipeline_version':PIPELINE_VERSION,'model':model.identity(),'source':base.source_hash(root/'src/vector_graph_rag_bilingualism'),
        'credential_source':{'file':str(args.model_env_file.resolve()),'variable':'VGRAG_CHAT_KEY'},
        'runner_sha256':base.sha(__file__),'adapter_sha256':base.sha(root/'evaluation/compare_light.py'),
        'judge_prompt_sha256':before['judge_prompt_sha256'],'role_caps':before['role_caps'],
        'workers':args.workers,'qwen_input_pacing_tpm':70000,'jev_http_concurrency':0,
        'comparison_scope':'Same frozen questions, snapshot, models and prompts as Jev run; Jev disabled'}
    target=args.output/'manifest.json'
    if target.exists() and json.loads(target.read_text())==manifest: pass
    base.atomic(target,manifest)
    if (args.output/'questions.jsonl').exists(): assert base.sha(args.output/'questions.jsonl')==manifest['sample_sha256']
    else: shutil.copy2(args.historical/'questions.jsonl',args.output/'questions.jsonl')
    pipeline=ChinesePipeline(store,stored['namespace'],embedder,model,config=config,snapshot=snapshot,
        count_tokens=lambda text:len(tokenizer.encode(text,add_special_tokens=False).ids),tokenizer_id='bge-small-zh-v1.5-frozen-tokenizer')
    pipeline.sync_milvus()
    latest={r['id']:r for r in base.rows(args.output/'generations.jsonl')}
    grades={r['id']:r for r in base.rows(args.output/'grades.jsonl') if r.get('grade')}
    def work(q):
        base.LOCAL.qid=q['id'];base.LOCAL.arm='no_jev'
        row=latest.get(q['id'])
        if row is None:
            started=time.monotonic()
            try:
                result=pipeline.query(q['question']);value=result.as_dict();stats=diagnostics(q,value,records)
                gold=set(q['sources']);candidate=stats['candidate_sources'] or {'recall':0,'all':False,'hit':[]};final=stats['final_sources']
                row={'id':q['id'],'arm':'no_jev','question':q['question'],'answer':result.answer,'status':result.status,
                     'citations':[{'id':p.id,'source':p.source,'text':p.text} for p in result.passages],
                     'candidate_recall':candidate['recall'],'candidate_all_sources':candidate['all'],
                     'candidate_matched_sources':candidate['hit'],'final_recall':final['recall'],'all_sources':final['all'],
                     'final_matched_sources':final['hit'],'recall_at_5':len(gold&{p.source for p in result.passages[:5]})/len(gold),
                     'trace':value['trace'],'evidence':value['evidence'],'missing_slots':result.missing_slots,'snapshot_id':snapshot}
            except Exception as error:
                row={'id':q['id'],'arm':'no_jev','question':q['question'],'answer':None,'status':'query_error',
                     'error':type(error).__name__,'citations':[],'candidate_recall':0,'candidate_all_sources':False,
                     'final_recall':0,'all_sources':False,'recall_at_5':0}
            row.update(documents=len(q['sources']),query_ms=round((time.monotonic()-started)*1000))
            base.append(args.output/'generations.jsonl',row)
            print(f"Generated {q['id']}: {row['status']}, final sources={row['final_recall']:.2f}, {row['query_ms']/1000:.1f}s",flush=True)
        if q['id'] not in grades:
            if not (row.get('answer') or '').strip():
                grade={'accuracy':'wrong','grounded':False,'reason':'未生成可评答案；空答案按未答对处理。'}
                graded={'id':q['id'],'arm':'no_jev','grade':grade,'rule_applied':'empty_answer'}
            else:
                payload={'question':q['question'],'reference':q['reference'],'answer':row['answer'],'evidence':row['citations']}
                try:
                    answer,_=client.ask('judge',JUDGE_PROMPT,payload,model=base.JUDGE,json_mode=True,max_tokens=1000)
                    grade=json.loads(answer)
                    if grade.get('accuracy') not in ('correct','partial','wrong') or type(grade.get('grounded')) is not bool:
                        raise ValueError('Invalid grade')
                    graded={'id':q['id'],'arm':'no_jev','grade':grade}
                except Exception as error:
                    graded={'id':q['id'],'arm':'no_jev','grade':None,'error':type(error).__name__}
            base.append(args.output/'grades.jsonl',graded)
            print(f"Graded {q['id']}: {(graded['grade'] or {}).get('accuracy','failed')}",flush=True)
        return q['id']
    started=time.time()
    def state(status):
        generated={r['id'] for r in base.rows(args.output/'generations.jsonl')}
        graded={r['id'] for r in base.rows(args.output/'grades.jsonl') if r.get('grade')}
        base.atomic(args.output/'run-state.json',{'status':status,'pid':os.getpid(),'updated_unix':time.time(),
            'process_started_unix':started,'target':1000,'generated':len(generated),'graded':len(graded)})
    try:
        print(f'Validated frozen snapshot {snapshot}; 100 questions (50 double + 50 triple); Jev disabled; workers={args.workers}.',flush=True)
        state('prepared' if args.prepare_only else 'running')
        if not args.prepare_only:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for task in as_completed([pool.submit(work,q) for q in sample]):
                    print('Completed: '+task.result(),flush=True);state('running')
            # Same-parameter retries only for failed grading, never repeat a valid grade or generation.
            for retry in range(2):
                latest={r['id']:r for r in base.rows(args.output/'generations.jsonl')}
                grades={r['id']:r for r in base.rows(args.output/'grades.jsonl') if r.get('grade')}
                pending=[q for q in sample if q['id'] in latest and q['id'] not in grades]
                if not pending: break
                print(f'Retry missing grades pass {retry+1}: {len(pending)}',flush=True)
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for task in as_completed([pool.submit(work,q) for q in pending]):
                        print('Completed: '+task.result(),flush=True);state('running')
        store.load(snapshot)
        assert base.sha(args.output/'questions.jsonl')==manifest['sample_sha256']
        assert all(base.sha(args.previous/n)==v for n,v in previous_hashes.items())
        assert base.source_hash(root/'src/vector_graph_rag_bilingualism')==manifest['source']
        base.atomic(args.output/'integrity-after.json',{'verified':True,'snapshot':snapshot,'index_content_hash':stored['content_hash'],
            'previous_files_unchanged':True,'historical_files_unchanged':True,'query_source_unchanged':True})
        complete=len({r['id'] for r in base.rows(args.output/'grades.jsonl') if r.get('grade')})==1000
        state('prepared' if args.prepare_only else 'complete' if complete else 'incomplete')
    except BaseException:
        state('interrupted');raise
    finally:
        client.http.close()


if __name__=='__main__': main()
