import json,hashlib,argparse,difflib
from pathlib import Path
from vector_graph_rag_bilingualism.storage import SnapshotStore
ap=argparse.ArgumentParser(description="Offline six-question Jev failure review; never calls a model")
ap.add_argument('--run',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--documents',type=Path,required=True)
a=ap.parse_args();run=a.run;out=a.output
if out.resolve()==run.resolve():raise ValueError('Use a separate audit output directory')
out.mkdir(parents=True,exist_ok=True)
input_names=['generations.jsonl','grades.jsonl','api-usage.jsonl','jev-http.jsonl','model-responses.jsonl','manifest.json','questions.jsonl']
initial_hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in input_names}
original={r['source']:r['text'] for r in map(json.loads,a.documents.read_text().splitlines())}
original_sha=hashlib.sha256(a.documents.read_bytes()).hexdigest()

def rows(name):return [json.loads(l) for l in (run/name).read_text().splitlines() if l.strip()]
qs={r['id']:r for r in rows('questions.jsonl')};gr={r['id']:r['grade'] for r in rows('grades.jsonl') if r.get('grade')}
m=json.loads((run/'manifest.json').read_text());_,records=SnapshotStore(run/'new-index').load(m['new_snapshot']);ps={r['passage']['id']:r['passage'] for r in records}
summary=[]
for r in rows('generations.jsonl'):
 qid=r['id']
 if gr[qid]['accuracy']=='correct':continue
 q=qs[qid];gold=set(q['sources']);ev=r['trace']['events'];goldps=[p for p in ps.values() if p['source'] in gold]
 fusions=[e['ids'] for e in ev if e['stage']=='fusion'];ranks=[e['ids'] for e in ev if e['stage']=='passage_rerank' and e.get('status')=='success'];selections=[e['evidence'] for e in ev if e['stage']=='selection' and 'evidence' in e];cov=[e for e in ev if e['stage']=='coverage'];final={p['id'] for p in r['citations']}
 flow=[]
 for p in goldps:
  pid=p['id'];flow.append({'id':pid,'source':p['source'],'title':p.get('metadata',{}).get('title'),'fusion_ranks':[a.index(pid)+1 if pid in a else None for a in fusions],'jev_ranks':[a.index(pid)+1 if pid in a else None for a in ranks],'selected_quotes':[[e for e in a if e['passage_id']==pid] for a in selections],'coverage_rounds':[i+1 for i,e in enumerate(cov) if pid in e['selected']],'budget_skipped':any(e['stage']=='context_budget' and e['skipped']==pid for e in ev),'final':pid in final,'text':p['text']})
 data={'question':q,'grade':gr[qid],'flow':flow,'answer':r['answer'],'evidence':r['evidence'],'citations':r['citations'],'events':[e for e in ev if e['stage'] in ['plan','coverage','verification','generation','answer_attempt','answer_repair','repair','context_budget','selection_fallback','index_health']], 'raw_model_responses':[e for e in rows('model-responses.jsonl') if e['question_id']==qid], 'api_errors':[e for e in rows('api-usage.jsonl') if e['question_id']==qid and (e['status']=='failed' or e.get('finish_reason') not in ('stop',None))]}
 (out/(qid.replace(':','-')+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
 summary.append({'id':qid,'grade':gr[qid]['accuracy'],'gold_sources':len(gold),'fused_sources':len({ps[pid]['source'] for a in fusions for pid in a}&gold),'ranked_sources':len({ps[pid]['source'] for a in ranks for pid in a}&gold),'initial_selected_sources':len({ps[e['passage_id']]['source'] for e in selections[0]}&gold),'selected_sources':len({ps[e['passage_id']]['source'] for a in selections for e in a}&gold),'final_sources':len({ps[pid]['source'] for pid in final}&gold),'final_passages':len(final),'coverage_units':[e['context_units'] for e in cov],'coverage_missing':[e['missing'] for e in cov],'rounds':len(ranks),'gold_flow':[{k:v for k,v in f.items() if k not in ['text','selected_quotes','title']} for f in flow]})
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps([{k:v for k,v in r.items() if k!='gold_flow'} for r in summary],ensure_ascii=False,indent=2))

checks=[]
for q in summary:
 for flow in q['gold_flow']:
  pid=flow['id'];source=flow['source'];before=original[source];after=ps[pid]['text']
  changes=[]
  for tag,i,j,k,l in difflib.SequenceMatcher(None,before,after).get_opcodes():
   if tag!='equal':changes.append({'type':tag,'original_offset':i,'original':before[i:j],'indexed':after[k:l]})
  checks.append({'question_id':q['id'],'source':source,'passage':pid,'exact_original_text':before==after,
                 'equal_ignoring_newlines':before.replace('\n','')==after.replace('\n',''),'differences':changes})
assert len(summary)==6 and len(checks)==16
assert all(c['equal_ignoring_newlines'] for c in checks)
assert all(q['fused_sources']==q['ranked_sources']==q['gold_sources'] for q in summary)
assert all(hashlib.sha256((run/n).read_bytes()).hexdigest()==v for n,v in initial_hashes.items())
assert hashlib.sha256(a.documents.read_bytes()).hexdigest()==original_sha
(out/'original-text-checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
(out/'read-only-check.json').write_text(json.dumps({'verified':True,'input_hashes':initial_hashes,
 'original_documents_sha256':original_sha,'questions_reviewed':6,'gold_texts_reviewed':16,
 'new_model_calls':0,'original_grades_changed':False},ensure_ascii=False,indent=2))
