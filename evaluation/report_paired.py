"""Offline paired statistics and review export; never invokes a model."""
import argparse
import csv
import json
import math
from pathlib import Path
import random


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=args.directory
    questions={q['id']:q for q in rows(root/'questions.jsonl')}
    results={(r['arm'],r['id']):r for r in rows(root/'generations.jsonl')}
    grades={(r['arm'],r['id']):r['grade'] for r in rows(root/'grades.jsonl') if r.get('grade')}
    paired=[]
    signed=[]
    for qid,q in questions.items():
        if not all((arm,qid) in grades for arm in ('light','bilingualism')):
            continue
        a,b=grades['light',qid],grades['bilingualism',qid]
        ra,rb=results['light',qid],results['bilingualism',qid]
        delta=int(b['accuracy']=='correct')-int(a['accuracy']=='correct')
        signed.append(delta)
        paired.append({'id':qid,'documents':len(q['sources']),'question':q['question'],
                       'light_grade':a['accuracy'],'new_grade':b['accuracy'],'delta':delta,
                       'light_candidate_recall':ra['candidate_recall'],'new_candidate_recall':rb['candidate_recall'],
                       'light_r5':ra['recall_at_5'],'new_r5':rb['recall_at_5'],
                       'light_final_recall':ra['final_recall'],'new_final_recall':rb['final_recall'],
                       'light_reason':a['reason'],'new_reason':b['reason'],
                       'light_status':ra['status'],'new_status':rb['status'],
                       'light_answer':ra.get('answer'),'new_answer':rb.get('answer'),
                       'reference':q['reference']})
    if not signed:
        raise ValueError('No paired graded results')
    rng=random.Random(2026111202)
    samples=sorted(sum(rng.choices(signed,k=len(signed)))/len(signed) for _ in range(20000))
    improved=signed.count(1); regressed=signed.count(-1); discordant=improved+regressed
    p=min(1.0,2*sum(math.comb(discordant,k) for k in range(min(improved,regressed)+1))/2**discordant) if discordant else 1.0
    statistics={'paired_questions':len(signed),'difference':sum(signed)/len(signed),
                'bootstrap_95_percentile':[samples[499],samples[19499]],'bootstrap_draws':20000,
                'improved':improved,'regressed':regressed,'mcnemar_exact_two_sided_p':p}
    (root/'paired-statistics.json').write_text(json.dumps(statistics,ensure_ascii=False,indent=2)+'\n')
    (root/'paired-review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired))
    with (root/'paired-review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]))
        writer.writeheader();writer.writerows(paired)
    print(json.dumps(statistics,ensure_ascii=False))


if __name__=='__main__':
    main()
