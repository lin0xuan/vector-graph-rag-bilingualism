"""Summarize observed runtime behavior without regrading or extra model calls."""
import json
from pathlib import Path
from collections import Counter
import sys

root=Path(sys.argv[1])
read=lambda name:[json.loads(line) for line in (root/name).read_text().splitlines() if line.strip()]
generations={(r['arm'],r['id']):r for r in read('generations.jsonl')}
counts=Counter(); cases=[]
for (arm,qid),row in generations.items():
    if arm!='bilingualism': continue
    trace=row.get('trace',{}).get('events',[])
    flags=[]
    verify=[e for e in trace if e['stage']=='verification']
    if any('numeric_copy_mismatch' in str(e.get('issues',[])) for e in verify):
        flags.append('numeric_rule_triggered')
    if any('missing_or_unknown_citation' in e.get('issues',[]) for e in verify):
        flags.append('citation_rule_triggered')
    if any(e['stage']=='answer_repair' for e in trace): flags.append('answer_repaired')
    if any(e['stage']=='repair' for e in trace): flags.append('retrieval_repair_attempted')
    if any(e['stage']=='repair' and not e['added'] for e in trace): flags.append('repair_without_new_candidates')
    if row.get('candidate_all_sources') and not row.get('all_sources'): flags.append('sources_lost_in_selection')
    if not row.get('candidate_all_sources'): flags.append('candidate_sources_missing')
    if row.get('missing_slots'): flags.append('missing_question_slots')
    for event in trace:
        if event.get('status') in ('failed','fallback'):
            flags.append(event['stage']+'_'+event['status'])
    counts.update(set(flags))
    cases.append({'id':qid,'flags':flags,'missing_slots':row.get('missing_slots',[]),
                  'verification':verify,'answer_repair':[e for e in trace if e['stage']=='answer_repair'],
                  'final_answer':row.get('answer')})
usage=read('api-usage.jsonl')
seen=Counter((r['arm'],r['question_id'],r['role'],r['request_hash']) for r in usage if r['status']=='complete')
output={'new_pipeline_observations':dict(counts),
        'repeated_identical_successful_requests':sum(n-1 for n in seen.values() if n>1),
        'non_stop_outputs':[{'arm':r['arm'],'id':r['question_id'],'role':r['role'],'finish':r.get('finish_reason')}
                            for r in usage if r['status']=='complete' and r.get('finish_reason')!='stop'],
        'cases':cases}
(root/'runtime-diagnosis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k!='cases'},ensure_ascii=False))
