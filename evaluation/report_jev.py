"""Offline audit of the frozen 20-question Jev / no-Jev pair."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
from report_repairs import rows, flags
from vector_graph_rag_bilingualism.storage import SnapshotStore


def qwen_usage(path):
    totals = defaultdict(Counter)
    for event in rows(path):
        group = 'judge' if event['role']=='judge' else 'query'
        value = totals[group]
        # A finished-but-invalid completion is followed by a logical error log.
        if event['status']=='complete' or 'finish_reason' not in event:
            value['http_attempts'] += 1
        if event['status']=='complete':
            usage = event.get('usage') or {}
            for key in ('prompt_tokens','completion_tokens','total_tokens'):
                value[key] += usage.get(key,0)
            value['usage_responses'] += bool(usage)
            value['large_context_responses'] += usage.get('prompt_tokens',0)>=128000
        if event['status']=='failed': value['failure_events'] += 1
    return {k:dict(v) for k,v in totals.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
    root=args.directory;manifest=json.loads((root/'manifest.json').read_text());control=Path(manifest['control'])
    for folder,key in [(control,'control_files'),(Path(manifest['baseline']),'baseline_files')]:
        for name,digest in manifest[key].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    assert json.loads((root/'integrity-after.json').read_text())['verified']
    questions={q['id']:q for q in rows(root/'questions.jsonl')};assert len(questions)==20
    _,records=SnapshotStore(root/'new-index').load(manifest['new_snapshot'])
    passages={r['passage']['id']:r['passage'] for r in records}
    outputs={};grades={};summary={};diagnosis={};review=[]
    for arm,folder in [('without',control),('jevhub',root)]:
        outputs[arm]={r['id']:r for r in rows(folder/'generations.jsonl')}
        grades[arm]={r['id']:r['grade'] for r in rows(folder/'grades.jsonl') if r.get('grade')}
        assert set(outputs[arm])==set(grades[arm])==set(questions)
        for qid,row in outputs[arm].items():
            gold=set(questions[qid]['sources']);citations=row['citations']
            final={p['source'] for p in citations}
            assert abs(len(final&gold)/len(gold)-row['final_recall'])<1e-12
            assert abs(len({p['source'] for p in citations[:5]}&gold)/len(gold)-row['recall_at_5'])<1e-12
            candidate=set()
            for event in row.get('trace',{}).get('events',[]):
                if event['stage']=='fusion': candidate.update(event['ids'])
                if event['stage']=='repair': candidate.update(event['added'])
            assert abs(len({passages[p]['source'] for p in candidate}&gold)/len(gold)-row['candidate_recall'])<1e-12
            for p in citations:
                assert p['source']==passages[p['id']]['source'] and p['text']==passages[p['id']]['text']
            for e in row.get('evidence',[]):
                assert passages[e['passage_id']]['text'][e['start']:e['end']]==e['quote']
            if not (row.get('answer') or '').strip():assert grades[arm][qid]['accuracy']=='wrong'
        summary[arm]={}
        for size in [0,2,3]:
            ids=[qid for qid,q in questions.items() if not size or len(q['sources'])==size]
            data=[outputs[arm][qid] for qid in ids]
            summary[arm][str(size or 'overall')]={'questions':len(ids),
                'grades':dict(Counter(grades[arm][qid]['accuracy'] for qid in ids)),
                **{k:statistics.mean(r[k] for r in data) for k in ['candidate_recall','final_recall','recall_at_5','query_ms']},
                'all_sources':sum(r['all_sources'] for r in data),
                'runtime_statuses':dict(Counter(r['status'] for r in data))}
        counts=Counter();fallbacks=[]
        for qid,row in outputs[arm].items():
            counts.update({k:int(v) for k,v in flags(row).items()})
            events=row.get('trace',{}).get('events',[])
            for stage in ['relation_rerank','passage_rerank']:
                counts[stage+'_success_questions']+=any(e['stage']==stage and e.get('status')=='success' for e in events)
                counts[stage+'_fallback_questions']+=any(e['stage']==stage and e.get('status')=='fallback' for e in events)
            for e in events:
                if e.get('status') in ('failed','fallback'):
                    fallbacks.append({'id':qid,**e})
        diagnosis[arm]={'question_counts':dict(counts),'fallback_events':fallbacks}
    changes={qid:int(grades['jevhub'][qid]['accuracy']=='correct')-int(grades['without'][qid]['accuracy']=='correct') for qid in questions}
    rng=random.Random(2026111202);boot=sorted(sum(rng.choices(list(changes.values()),k=20))/20 for _ in range(20000))
    pairs={'improved':[qid for qid,v in changes.items() if v==1], 'regressed':[qid for qid,v in changes.items() if v==-1],
           'difference':statistics.mean(changes.values()),'bootstrap_95_percentile':[boot[499],boot[19499]]}
    jev_costs={}
    events=rows(root/'jev-http.jsonl')
    for phase in sorted({e['phase'] for e in events}):
        data=[e for e in events if e['phase']==phase]
        jev_costs[phase]={'http_attempts':len(data),'statuses':dict(Counter(str(e.get('http_status','transport_error')) for e in data)),
            'reported_charge_usd':str(sum((Decimal(e['charge_usd']) for e in data if e['charge_usd'] is not None),Decimal(0))),
            'charge_unknown_attempts':sum(e['charge_usd'] is None for e in data),
            'usage_unknown_attempts':sum(e['usage'] is None for e in data),
            'response_models':dict(Counter(e.get('response_model','unknown') for e in data)),
            'stages':dict(Counter(e['stage'] for e in data))}
    costs={'without':qwen_usage(control/'api-usage.jsonl'),'jevhub':qwen_usage(root/'api-usage.jsonl')}
    for qid,q in questions.items():
        row={'id':qid,'documents':len(q['sources']),'question':q['question'],'reference':q['reference']}
        for arm in ['without','jevhub']:
            row.update({arm+'_'+k:outputs[arm][qid].get(k) for k in ['candidate_recall','final_recall','recall_at_5','answer','status']})
            row[arm+'_grade']=grades[arm][qid]['accuracy'];row[arm+'_reason']=grades[arm][qid].get('reason','')
        review.append(row)
    with (root/'paired-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(review[0]));writer.writeheader();writer.writerows(review)
    # Public price estimate for returned usage only, never a verified invoice.
    estimates={}
    for arm,data in costs.items():
        estimates[arm]={}
        for phase,entry in data.items():
            if entry.get('large_context_responses'):
                estimates[arm][phase]=None
                continue
            ip,op=(Decimal('1.8'),Decimal('10.8')) if phase=='judge' else (Decimal('0.4'),Decimal('3.2'))
            estimates[arm][phase]=str((Decimal(entry.get('prompt_tokens',0))*ip+Decimal(entry.get('completion_tokens',0))*op)/Decimal(1000000))
    invalid_refs={arm:[qid for qid,row in data.items() if set(re.findall(r'\[([^\[\]]+)\]',row.get('answer') or ''))-{p['id'] for p in row['citations']}] for arm,data in outputs.items()}
    qwen_errors=dict(Counter(e.get('error') for e in rows(root/'api-usage.jsonl') if e['status']=='failed'))
    result={'invalid_final_citation_questions':invalid_refs,'qwen_error_events':qwen_errors,'qwen_public_price_estimate_cny':estimates,'price_source':'https://cloud-rd.siliconflow.cn/pricing','price_checked':'2026-11-12','summary':summary,'pairs':pairs,'diagnosis':diagnosis,'jev_gateway_costs':jev_costs,'qwen_usage':costs,
            'verified_pairs':20,'snapshot':manifest['new_snapshot'],'invoice_verified':False}
    (root/'paired-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    pct=lambda value:f'{value*100:.2f}%'
    lines=['# 中文 20 题：启用 JevHub 前后对照','','日期：2026-11-12。未启用组复用正式修复基线；启用组仅重新运行原 20 题，没有扩题或跑全量。',
           '', '| 指标 | 未启用 Jev | 启用 Jev（JevHub） |','|---|---:|---:|']
    for size,label in [('overall','完整答对'),('2','双文档完整答对'),('3','三文档完整答对')]:
        cells=[]
        for arm in ['without','jevhub']:
            s=summary[arm][size];n=s['questions'];c=s['grades'].get('correct',0);cells.append(f'{c}/{n}（{pct(c/n)}）')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    for key,label in [('candidate_recall','候选来源召回'),('final_recall','最终证据来源召回'),('recall_at_5','最终证据 Recall@5')]:
        lines.append('| '+label+' | '+' | '.join(pct(summary[a]['overall'][key]) for a in ['without','jevhub'])+' |')
    lines.append('| 找齐全部标准来源 | '+' | '.join(str(summary[a]['overall']['all_sources'])+'/20' for a in ['without','jevhub'])+' |')
    lines+=['',f"完整答对变化：{pairs['difference']*100:+.2f} 个百分点。救回：{', '.join(pairs['improved']) or '无'}；退化：{', '.join(pairs['regressed']) or '无'}。",
            f"20,000 次配对重采样 95% 区间：[{boot[499]*100:+.2f}, {boot[19499]*100:+.2f}] 个百分点。",'',
            '来源召回是标准来源 ID 命中比例的逐题平均，不等于关键事实召回；完整答对沿用同一机器裁判，运行状态不代表正确率。20 题是开发回归样本，且两组请求时间不同，不能外推千题总体准确率或排除服务波动。',
            '', '## 控制条件', '',
            f"全部 2,309 份资料、2,993 段、BGE-small 向量及关系快照保持不变：`{manifest['new_snapshot']}`。",
            '问题、Qwen/Qwen3.5-35B-A3B 回答模型、Qwen/Qwen3.6-35B-A3B 评分模型、温度 0、关闭思考、回答上限 2,000 token、提示词、候选池和补查预算与修复基线相同。只新增 Jev 关系/段落评分；保留后续中文选证、回答和核验。',
            '请求固定 jev-1.13.0，通过第三方 JevHub 调用；报告记录其实际响应模型字段，未独立验证供应商内部路由。模型列表只列动态别名，但显式 1.13.0 的真实双阶段测试成功。未调用 latest 替代固定版本。',
            '离线复算核对了 40 份配对输出、来源召回、证据原文偏移和索引内容；旧基线文件及运行期源代码指纹均未改变。','',
            '## 费用和用量','','网关金额来自每个响应的 X-JevHub-Charge-USD，使用 Decimal 累加；不是官方单价估算，也不是独立核对的账户账单。网关不返回 usage，不能把未知 token 用量写成 0。',
            '', '| 阶段 | HTTP 尝试 | 网关报告扣费（美元） | 缺少扣费头 |','|---|---:|---:|---:|']
    for phase,c in jev_costs.items():lines.append(f"| {phase} | {c['http_attempts']} | {c['reported_charge_usd']} | {c['charge_unknown_attempts']} |")
    lines+=['','Qwen 回答/选证/核验与评分费用另计。下表仅汇总服务成功返回的 token 计量；超时可能消耗未返回的用量，不作为完整账单。',
            '', '| 组/阶段 | HTTP 尝试 | 输入 token | 输出 token |','|---|---:|---:|---:|']
    for arm,data in costs.items():
        for phase,c in data.items():lines.append(f"| {arm}/{phase} | {c.get('http_attempts',0)} | {c.get('prompt_tokens',0)} | {c.get('completion_tokens',0)} |")
    lines+=['','按 2026-11-12 查阅的[硅基流动公开价格页](https://cloud-rd.siliconflow.cn/pricing)，Qwen3.5-35B-A3B 在输入小于 128k 时为输入 0.40 / 输出 3.20 元每百万 token；Qwen3.6-35B-A3B 为输入 1.80 / 输出 10.80 元每百万 token。仅按已返回用量估算，不含未知超时用量、套餐折扣或账户账单核验。']
    lines.append('')
    for arm,entry in estimates.items():
        lines.append(f"- {arm}：查询阶段约 ¥{entry.get('query')}，评分阶段约 ¥{entry.get('judge')}。")
    lines+=['','## 回退记录','','| 触发题数 | 未启用 | JevHub |','|---|---:|---:|']
    for key,label in [('relation_rerank_fallback_questions','关系重排回退'),('passage_rerank_fallback_questions','段落重排回退'),('selection_fallback','选证失败保留原文'),('answer_fallback','生成/核验异常保留答案'),('verification_retrieval','核验缺证后补查')]:
        lines.append('| '+label+' | '+' | '.join(str(diagnosis[a]['question_counts'].get(key,0)) for a in ['without','jevhub'])+' |')
    lines+=['',f"最终引用 ID 不匹配题数：未启用 {len(invalid_refs['without'])}，JevHub {len(invalid_refs['jevhub'])}。本轮中间核验触发数字规则 3 题、引用规则 1 题；这不是自动认定的误报数，也不能宣称规则已无误报。",f"Qwen 错误事件：{json.dumps(qwen_errors,ensure_ascii=False)}。网关发生 1 次连接超时，同一请求重试成功；正式 20 题均有成功的关系和段落 Jev 评分。",'2docs:473 首次评分达到 1,000 token 上限而截断；随后以相同模型、提示和上限补评分成功。文件哈希确认未重新生成任何答案或调用 Jev，21 次评分请求费用均已计入。']
    lines+=['','具体错误和题号保存在 paired-summary.json 的 diagnosis；逐题答案、裁判理由见 paired-review.csv。','',
            '本机材料目录：`.eval/crud-jevhub-20-20261112/`（Git 忽略）。正式入口 `evaluation/retest_jev.py`，离线报告 `evaluation/report_jev.py`。原官方地址 401 验证单独保留于 `.eval/crud-jev-20-20261112/`；此次网关预检和评分未重复使用该错误目录。','']
    if (root/'case-review.md').exists():lines.append((root/'case-review.md').read_text())
    args.report.write_text('\n'.join(lines))
    print(json.dumps({'summary':summary,'pairs':pairs,'jev_costs':jev_costs,'qwen_usage':costs},ensure_ascii=False))


if __name__=='__main__':main()
