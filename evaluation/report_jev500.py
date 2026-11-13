"""Offline interim/final report of a fixed Jev Chinese multi-hop run; no API calls."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import statistics


def read_rows(path):
    if not path.exists(): return []
    data=path.read_bytes(); lines=data.splitlines(); result=[]
    for i,line in enumerate(lines):
        if not line.strip(): continue
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not data.endswith(b'\n'): break
            raise
    return result


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temp=path.with_suffix(path.suffix+'.report-tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def pct(v): return '—' if v is None else f'{100*v:.2f}%'
def mean(data,key): return statistics.mean(r[key] for r in data) if data else None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--historical',type=Path);ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args();root=args.directory;out=args.output_dir or root;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    assert sha(root/'questions.jsonl')==manifest['sample_sha256'],'Sample changed'
    qs={q['id']:q for q in read_rows(root/'questions.jsonl')}
    generations={r['id']:r for r in read_rows(root/'generations.jsonl')}
    grades={r['id']:r['grade'] for r in read_rows(root/'grades.jsonl') if r.get('grade')}
    assert set(generations)<=set(qs) and set(grades)<=set(generations),'Unknown IDs'
    from vector_graph_rag_bilingualism.storage import SnapshotStore
    stored,records=SnapshotStore(root/'new-index').load(manifest['new_snapshot'])
    if manifest.get('index_content_hash'): assert stored['content_hash']==manifest['index_content_hash']
    passages={r['passage']['id']:r['passage'] for r in records}
    invalid=[];fallback=Counter();fallback_events=[];diagnosis=defaultdict(list)
    for qid,row in generations.items():
        gold=set(qs[qid]['sources']);citations=row.get('citations',[]);final={p['source'] for p in citations}
        assert abs(len(final&gold)/len(gold)-row['final_recall'])<1e-12
        assert abs(len({p['source'] for p in citations[:5]}&gold)/len(gold)-row['recall_at_5'])<1e-12
        assert row['all_sources']==(gold<=final)
        ids=set();events=row.get('trace',{}).get('events',[])
        for event in events:
            if event['stage']=='fusion':ids.update(event['ids'])
            if event['stage']=='repair':ids.update(event['added'])
            if event.get('status') in ('failed','fallback'):fallback_events.append({'id':qid,**event})
        assert abs(len({passages[p]['source'] for p in ids}&gold)/len(gold)-row['candidate_recall'])<1e-12
        for p in citations:assert p['source']==passages[p['id']]['source'] and p['text']==passages[p['id']]['text']
        for e in row.get('evidence',[]):assert passages[e['passage_id']]['text'][e['start']:e['end']]==e['quote']
        if qid in grades:
            assert grades[qid]['accuracy'] in ('correct','partial','wrong')
            if not (row.get('answer') or '').strip():assert grades[qid]['accuracy']=='wrong'
        bad=set(re.findall(r'\[([^\[\]]+)\]',row.get('answer') or ''))-{p['id'] for p in citations}
        if bad:invalid.append({'id':qid,'refs':sorted(bad)})
        names=['selection_fallback','answer_fallback','answer_repair']
        for name in names:
            if any(e['stage']==name for e in events):fallback[name]+=1
        for stage in ['relation_rerank','passage_rerank']:
            for status in ['success','fallback']:
                if any(e['stage']==stage and e.get('status')==status for e in events):fallback[stage+'_'+status]+=1
        if any(e['stage']=='repair' and e.get('reason')=='verification_gap' for e in events):fallback['verification_retrieval']+=1
        if row['candidate_all_sources'] and not row['all_sources']:diagnosis['candidate_complete_final_missing'].append(qid)
        if qid in grades and grades[qid]['accuracy']!='correct':
            key='not_correct_final_sources_complete' if row['all_sources'] else 'not_correct_final_sources_missing'
            diagnosis[key].append(qid)
    summary={}
    for size in [0,2,3]:
        ids={qid for qid,q in qs.items() if not size or len(q['sources'])==size};done=ids&set(generations);scored=ids&set(grades)
        data=[generations[qid] for qid in sorted(done)];counts=Counter(grades[qid]['accuracy'] for qid in scored)
        summary[str(size or 'overall')]={'questions':len(ids),'generated':len(done),'graded':len(scored),
            'pending_generation':len(ids-done),'pending_grade':len(done-scored),'grades':dict(counts),
            'complete_accuracy':counts['correct']/len(ids) if len(scored)==len(ids) and ids else None,
            'provisional_accuracy_among_graded':counts['correct']/len(scored) if scored else None,
            **{key:mean(data,key) for key in ['candidate_recall','final_recall','recall_at_5','query_ms']},
            'candidate_all_sources':sum(r['candidate_all_sources'] for r in data),'all_sources':sum(r['all_sources'] for r in data),
            'runtime_statuses':dict(Counter(r['status'] for r in data))}
    qwen=defaultdict(Counter)
    for e in read_rows(root/'api-usage.jsonl'):
        key='judge' if e['role']=='judge' else 'query';c=qwen[key]
        if e['status']=='complete' or 'finish_reason' not in e:c['http_attempts']+=1
        if e['status']=='complete':
            usage=e.get('usage') or {};c['responses_with_usage']+=bool(usage);c['responses_missing_usage']+=not bool(usage)
            for k in ['prompt_tokens','completion_tokens','total_tokens']:c[k]+=usage.get(k,0)
        if e['status']=='failed':c['failure_events']+=1
    http=read_rows(root/'jev-http.jsonl');jev={'http_attempts':len(http),
        'reported_charge_usd':str(sum((Decimal(e['charge_usd']) for e in http if e.get('charge_usd') is not None),Decimal(0))),
        'charge_unknown_attempts':sum(e.get('charge_usd') is None for e in http),
        'statuses':dict(Counter(str(e.get('http_status','transport_error')) for e in http)),
        'response_models':dict(Counter(e.get('response_model','unknown') for e in http))}
    paired=None
    if args.historical:
        oldq={q['id']:q for q in read_rows(args.historical/'questions.jsonl')}
        old={r['id']:r for r in read_rows(args.historical/'results.jsonl') if r.get('grade')}
        assert set(qs)<=set(oldq) and set(qs)<=set(old)
        for qid,q in qs.items():assert all(q[k]==oldq[qid][k] for k in ['question','reference','sources'])
        paired={}
        for size in [0,2,3]:
            ids={qid for qid,q in qs.items() if (not size or len(q['sources'])==size)}&set(grades)
            improved=sorted(qid for qid in ids if grades[qid]['accuracy']=='correct' and old[qid]['grade']['accuracy']!='correct')
            regressed=sorted(qid for qid in ids if grades[qid]['accuracy']!='correct' and old[qid]['grade']['accuracy']=='correct')
            paired[str(size or 'overall')]={'paired_questions':len(ids),'old_correct':sum(old[i]['grade']['accuracy']=='correct' for i in ids),
                'new_correct':sum(grades[i]['accuracy']=='correct' for i in ids),'improved':improved,'regressed':regressed,
                'difference':(len(improved)-len(regressed))/len(ids) if ids else None}
    integrity_path=root/'integrity-after.json';integrity=json.loads(integrity_path.read_text()) if integrity_path.exists() else None
    complete=set(qs)==set(generations)==set(grades)
    result={'complete':complete,'summary':summary,'paired_historical_light':paired,'diagnosis':dict(diagnosis),
        'fallback_question_counts':dict(fallback),'fallback_events':fallback_events,'invalid_final_citations':invalid,
        'jev_gateway_cost':jev,'qwen_usage':dict(qwen),'snapshot':manifest['new_snapshot'],'index_content_verified':True,
        'integrity_after':integrity,'invoice_verified':False,'source_recall_is_not_fact_recall':True}
    save(out/'summary-500.json',result)
    review=[]
    for qid,q in qs.items():
        row=generations.get(qid,{})
        entry={'id':qid,'documents':len(q['sources']),'question':q['question'],'reference':q['reference'],
               **{k:row.get(k) for k in ['answer','status','candidate_recall','final_recall','recall_at_5','all_sources']},
               'grade':grades.get(qid,{}).get('accuracy'),'grade_reason':grades.get(qid,{}).get('reason')}
        if args.historical:entry.update(old_grade=old[qid]['grade']['accuracy'],old_reason=old[qid]['grade'].get('reason'))
        review.append(entry)
    with (out/'review-500.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(review[0]));writer.writeheader();writer.writerows(review)
    n=len(qs);lines=[f'# 中文 JevHub 多跳 {n} 题评测','',
        '状态：'+('全部生成与评分完成。' if complete else '运行中；以下已评分正确率是临时值，不代表整批最终成绩。'),'',
        '| 分组 | 计划 | 已生成 | 已评分 | correct / partial / wrong | 已评分完整答对率 |',
        '|---|---:|---:|---:|---|---:|']
    for key,label in [('overall','全部'),('2','双文档'),('3','三文档')]:
        r=summary[key];c=r['grades'];lines.append(f"| {label} | {r['questions']} | {r['generated']} | {r['graded']} | {c.get('correct',0)} / {c.get('partial',0)} / {c.get('wrong',0)} | {pct(r['provisional_accuracy_among_graded'])} |")
    lines+=['','| 分组 | 候选来源召回 | 最终来源召回 | Recall@5 | 找齐全部来源 / 已生成 |','|---|---:|---:|---:|---:|']
    for key,label in [('overall','全部'),('2','双文档'),('3','三文档')]:
        r=summary[key];lines.append(f"| {label} | {pct(r['candidate_recall'])} | {pct(r['final_recall'])} | {pct(r['recall_at_5'])} | {r['all_sources']}/{r['generated']} |")
    lines+=['','来源召回按题平均，不等于关键事实召回。完整答对沿用机器裁判原标签；不选择性重评存在歧义的答案。',
        '',f"冻结索引：`{manifest['new_snapshot']}`。本报告逐题复算候选和最终来源命中、原文内容及引文偏移；索引内容校验通过。",
        '运行期输入与源码一致性检查：'+('已通过。' if complete and integrity and integrity.get('verified') else '尚未完成；运行中仅能确认当前索引内容。'),'',
        '## 费用与用量','',f"Jev 网关响应头累计报告扣费 ${jev['reported_charge_usd']}；HTTP 尝试 {jev['http_attempts']} 次，其中 {jev['charge_unknown_attempts']} 次没有扣费计量。此金额未经账户账单核对。",'',
        '| Qwen 阶段 | HTTP 尝试 | 输入 token | 输出 token | 错误事件 |','|---|---:|---:|---:|---:|']
    for phase,c in qwen.items():lines.append(f"| {phase} | {c['http_attempts']} | {c['prompt_tokens']} | {c['completion_tokens']} | {c['failure_events']} |")
    lines+=['','Qwen 费用另计；这里只记录服务返回的 token 用量，不把超时未知用量记为零，也不套用未经重新核实的价格。',
            '', '## 回退与后续诊断','', '| 事件 | 涉及题数 |','|---|---:|']
    for key,value in sorted(fallback.items()):lines.append(f'| {key} | {value} |')
    lines +=['',f'最终引用 ID 不匹配 {len(invalid)} 题。候选来源齐全但最终缺来源 {len(diagnosis["candidate_complete_final_missing"])} 题；最终来源齐全但未完整答对 {len(diagnosis["not_correct_final_sources_complete"])} 题。这些标签只定位阶段，不能替代逐题关键事实复核。']
    if paired:
        lines+=['','## 同题历史轻量版参考','','历史轻量版是已有运行结果，此次没有重新调用。旧版为 single-stage，评分上限 500 token；本轮保留最近 20 题的双阶段 Jev 链路、评分上限 1000 token 和引用格式豁免提示。模型名称相同也不能保证服务版本一致；以下为整条链路历史同题对照，不能归因为 Jev 单因素增益。','','| 分组 | 配对题数 | 历史正确 | 本轮正确 | 正确率差 | 救回 / 退化 |','|---|---:|---:|---:|---:|---|']
        for key,label in [('overall','全部'),('2','双文档'),('3','三文档')]:
            p=paired[key];d='—' if p['difference'] is None else f"{100*p['difference']:+.2f} pp";lines.append(f"| {label} | {p['paired_questions']} | {p['old_correct']} | {p['new_correct']} | {d} | {len(p['improved'])} / {len(p['regressed'])} |")
    lines+=['','逐题答案、原始评分理由和历史标签见 `review-500.csv`；机器可读统计与回退事件见 `summary-500.json`。本报告脚本没有模型或网络调用。','']
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text('\n'.join(lines))
    print(json.dumps({'complete':complete,'summary':summary,'jev_gateway_cost':jev,'qwen_usage':dict(qwen)},ensure_ascii=False))

if __name__=='__main__':main()
