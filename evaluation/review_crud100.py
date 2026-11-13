import json,hashlib,re
from pathlib import Path
from collections import Counter
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from vector_graph_rag_bilingualism.storage import SnapshotStore
run=root/'.eval/crud-jevhub-100-20261112';out=root/'.eval/reviews/crud100-offline';out.mkdir(parents=True,exist_ok=True)
input_names=['generations.jsonl','grades.jsonl','model-responses.jsonl','api-usage.jsonl','jev-http.jsonl','questions.jsonl','manifest.json','new-index/catalog.sqlite3']
initial_hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in input_names}
def rows(n):return [json.loads(l) for l in (run/n).read_text().splitlines()]
qs={r['id']:r for r in rows('questions.jsonl')};gr={r['id']:r['grade'] for r in rows('grades.jsonl') if r.get('grade')};gen={r['id']:r for r in rows('generations.jsonl')}
old={r['id']:r for r in map(json.loads,Path('/Users/hootoom002/Documents/vector-graph-rag-chinese-light/.vgrag-eval/crud-500.lqeu5e/results.jsonl').read_text().splitlines()) if r.get('grade')}
m=json.loads((run/'manifest.json').read_text());_,records=SnapshotStore(run/'new-index').load(m['new_snapshot']);ps={r['passage']['id']:r['passage'] for r in records}
raw=rows('model-responses.jsonl');api=rows('api-usage.jsonl');summary=[]
for qid,r in gen.items():
 q=qs[qid];gold=set(q['sources']);ev=r.get('trace',{}).get('events',[])
 fusions=[e['ids'] for e in ev if e['stage']=='fusion'];ranks=[e['ids'] for e in ev if e['stage']=='passage_rerank' and e.get('status')=='success'];sel=[e['evidence'] for e in ev if e['stage']=='selection' and 'evidence' in e];cov=[e for e in ev if e['stage']=='coverage'];final={p['id'] for p in r['citations']}
 flow=[]
 for p in ps.values():
  if p['source'] not in gold:continue
  pid=p['id'];flow.append({'id':pid,'source':p['source'],'fusion_ranks':[a.index(pid)+1 if pid in a else None for a in fusions],'jev_ranks':[a.index(pid)+1 if pid in a else None for a in ranks],'selected_quotes':[[e['quote'] for e in a if e['passage_id']==pid] for a in sel],'coverage_rounds':[i+1 for i,e in enumerate(cov) if pid in e['selected']],'budget_skipped':any(e['stage']=='context_budget' and e['skipped']==pid for e in ev),'final':pid in final,'text':p['text']})
 def sources(ids):return len({ps[pid]['source'] for pid in ids}&gold)
 stat={'id':qid,'grade':gr[qid]['accuracy'],'old_grade':old[qid]['grade']['accuracy'],'gold':len(gold),'candidate':r['candidate_recall'],'final':r['final_recall'],'fused_sources':sources([pid for a in fusions for pid in a]),'ranked_sources':sources([pid for a in ranks for pid in a]),'initial_selected_sources':sources([e['passage_id'] for e in sel[0]]) if sel else 0,'selected_sources':sources([e['passage_id'] for a in sel for e in a]),'final_sources':sources(final),'final_passages':len(final),'units':[e['context_units'] for e in cov],'missing':[e['missing'] for e in cov],'fallbacks':[e for e in ev if e['stage'] in ['selection_fallback','answer_fallback']],'verification':[e for e in ev if e['stage']=='verification'],'budget_skipped_gold':sum(f['budget_skipped'] for f in flow),'status':r['status']}
 summary.append(stat)
 if gr[qid]['accuracy']!='correct':
  data={'question':q,'grade':gr[qid],'old_grade':old[qid]['grade'],'old_answer':old[qid]['answer'],'old_citations':old[qid]['citations'],'summary':stat,'flow':flow,'answer':r['answer'],'citations':r['citations'],'events':[e for e in ev if e['stage'] in ['plan','coverage','verification','generation','answer_attempt','answer_repair','repair','context_budget','selection_fallback','answer_fallback']], 'raw_model_responses':[e for e in raw if e['question_id']==qid], 'api_errors':[e for e in api if e['question_id']==qid and e['status']=='failed']}
  (out/(qid.replace(':','-')+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(out/'input-hashes.json').write_text(json.dumps({n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ['generations.jsonl','grades.jsonl','model-responses.jsonl','api-usage.jsonl','jev-http.jsonl','questions.jsonl','manifest.json']},indent=2))
for s in summary:
 if s['gold']==3 and s['grade']!='correct':print(json.dumps({k:v for k,v in s.items() if k not in ['verification','fallbacks','missing']},ensure_ascii=False))

original_path=Path('/Users/hootoom002/Documents/vector-graph-rag-chinese-light/.vgrag-eval/crud-1000-titled/documents.jsonl')
original={r['source']:r['text'] for r in map(json.loads,original_path.read_text().splitlines())}
checks=[]
for item in summary:
 if item['gold']!=3 or item['grade']=='correct':continue
 data=json.loads((out/(item['id'].replace(':','-')+'.json')).read_text())
 for flow in data['flow']:
  before=original[flow['source']];after=flow['text']
  checks.append({'question':item['id'],'source':flow['source'],'passage':flow['id'],
   'equal_ignoring_newlines':before.replace('\n','')==after.replace('\n',''),
   'equal_ignoring_whitespace':re.sub(r'\s+','',before)==re.sub(r'\s+','',after)})
assert len(checks)==72 and all(c['equal_ignoring_whitespace'] for c in checks)
assert all(hashlib.sha256((run/n).read_bytes()).hexdigest()==v for n,v in initial_hashes.items())
(out/'original-text-checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
(out/'read-only-check.json').write_text(json.dumps({'verified':True,'input_hashes':initial_hashes,
 'original_documents_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),
 'questions_stage_audited':100,'noncorrect_case_files':34,'triple_noncorrect':24,
 'triple_regressions':13,'new_model_calls':0,'original_grades_changed':False},ensure_ascii=False,indent=2))
