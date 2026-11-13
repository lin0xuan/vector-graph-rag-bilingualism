"""Offline three-arm comparison; fails closed on incomplete pairs or changed evidence."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def latest(path):
    return {(r['arm'],r['id']):r for r in rows(path)}


def flags(row):
    events=row.get('trace',{}).get('events',[])
    checks=[e for e in events if e['stage']=='verification']
    return {'numeric_rule_triggered':any('numeric_copy_mismatch' in str(e.get('issues')) for e in checks),
            'citation_rule_triggered':any('missing_or_unknown_citation' in str(e.get('issues')) for e in checks),
            'answer_repair':any(e['stage']=='answer_repair' for e in events),
            'answer_fallback':any(e['stage']=='answer_fallback' for e in events),
            'selection_failed':any(e['stage']=='selection' and e.get('status')=='failed' for e in events),
            'selection_fallback':any(e['stage']=='selection_fallback' for e in events),
            'verification_retrieval':any(e['stage']=='repair' and e.get('reason')=='verification_gap' for e in events),
            'sources_lost':row['candidate_all_sources'] and not row['all_sources']}


def usage(path,arm):
    out=Counter()
    for e in rows(path):
        if e['arm']!=arm or e['role']=='judge': continue
        if e['status']=='complete' or 'finish_reason' not in e: out['http_attempts']+=1
        if e['status']=='complete':
            for k in ['prompt_tokens','completion_tokens','total_tokens']:out[k]+=(e.get('usage') or {}).get(k,0)
        if e['status']=='failed':out['failure_events']+=1
    return dict(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
    root=args.directory;manifest=json.loads((root/'manifest.json').read_text());baseline=Path(manifest['baseline'])
    for name,checksum in manifest['baseline_files'].items():assert hashlib.sha256((baseline/name).read_bytes()).hexdigest()==checksum
    assert json.loads((root/'integrity-after.json').read_text())['verified']
    questions={q['id']:q for q in rows(root/'questions.jsonl')};assert len(questions)==20
    allrows={**latest(baseline/'generations.jsonl'),**latest(root/'generations.jsonl')}
    grades={(r['arm'],r['id']):r['grade'] for path in [baseline/'grades.jsonl',root/'grades.jsonl'] for r in rows(path) if r.get('grade')}
    arms=['light','bilingualism','fixed'];expected={(arm,qid) for arm in arms for qid in questions}
    assert set(allrows)==set(grades)==expected,'Incomplete 20-question / three-arm comparison'
    # Independent recomputation from source IDs and recorded candidate lists.
    from vector_graph_rag_bilingualism.storage import SnapshotStore
    store=SnapshotStore(root/'new-index');_,records=store.load(manifest['new_snapshot'])
    source={r['passage']['id']:r['passage']['source'] for r in records}
    for (arm,qid),row in allrows.items():
        gold=set(questions[qid]['sources']);final={p['source'] for p in row['citations']}
        assert abs(len(gold&final)/len(gold)-row['final_recall'])<1e-12
        assert abs(len(gold&{p['source'] for p in row['citations'][:5]})/len(gold)-row['recall_at_5'])<1e-12
        if arm=='light':candidate={p['source'] for p in row.get('candidates',[])}
        else:
            ids=set()
            for e in row.get('trace',{}).get('events',[]):
                if e['stage']=='fusion':ids.update(e['ids'])
                if e['stage']=='repair':ids.update(e['added'])
            candidate={source[i] for i in ids}
            passages={p['id']:p['text'] for p in row['citations']}
            for ev in row.get('evidence',[]):assert passages[ev['passage_id']][ev['start']:ev['end']]==ev['quote']
        assert abs(len(candidate&gold)/len(gold)-row['candidate_recall'])<1e-12
        if not (row.get('answer') or '').strip():assert grades[arm,qid]['accuracy']=='wrong'
    summary={}
    for group in [0,2,3]:
        summary[str(group or 'overall')]={}
        for arm in arms:
            groupids=[qid for qid,q in questions.items() if not group or len(q['sources'])==group]
            data=[allrows[arm,qid] for qid in groupids];counts=Counter(grades[arm,qid]['accuracy'] for qid in groupids)
            summary[str(group or 'overall')][arm]={'questions':len(data),'grades':dict(counts),
                **{key:statistics.mean(r[key] for r in data) for key in ['candidate_recall','final_recall','recall_at_5','query_ms']},
                'all_sources':sum(r['all_sources'] for r in data),'candidate_all_sources':sum(r['candidate_all_sources'] for r in data),
                'runtime_statuses':dict(Counter(r['status'] for r in data))}
    pairs={}
    for other in ['light','bilingualism']:
        changes={qid:int(grades['fixed',qid]['accuracy']=='correct')-int(grades[other,qid]['accuracy']=='correct') for qid in questions}
        rng=random.Random(2026111202);values=list(changes.values());boot=sorted(sum(rng.choices(values,k=20))/20 for _ in range(20000))
        pairs[other]={'improved':[qid for qid,v in changes.items() if v==1],'regressed':[qid for qid,v in changes.items() if v==-1],
                      'difference':statistics.mean(values),'bootstrap_95_percentile':[boot[499],boot[19499]]}
    diagnosis={arm:dict(sum((Counter({k:int(v) for k,v in flags(allrows[arm,qid]).items()}) for qid in questions),Counter())) for arm in ['bilingualism','fixed']}
    invalid={}
    for arm in ['bilingualism','fixed']:
        invalid[arm]=[]
        for qid in questions:
            row=allrows[arm,qid];allowed={p['id'] for p in row['citations']}
            refs=re.findall(r'\[([^\[\]]+)\]',row.get('answer') or '')
            bad=sorted(set(refs)-allowed)
            if bad:invalid[arm].append({'id':qid,'refs':bad})
    pilot_paths=[p for p in root.parent.glob('crud-repair-pilot*-20261112') if p.is_dir()]
    pilot_usage=dict(sum((Counter(usage(p/'api-usage.jsonl','fixed')) for p in pilot_paths),Counter()))
    pilot_generations=sum(len(rows(p/'generations.jsonl')) for p in pilot_paths if (p/'generations.jsonl').exists())
    costs={arm:usage(baseline/'api-usage.jsonl' if arm!='fixed' else root/'api-usage.jsonl',arm) for arm in arms}
    output={'summary':summary,'pairs':pairs,'diagnosis':diagnosis,'invalid_final_citations':invalid,'query_usage':costs,
            'pilot_query_usage':pilot_usage,'pilot_generations':pilot_generations,
            'verified_pairs':20,'verified_answers':60,'jev_enabled':manifest['jev_enabled']}
    (root/'three-arm-summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    review=[]
    for qid,q in questions.items():
        r={'id':qid,'documents':len(q['sources']),'question':q['question'],'reference':q['reference']}
        for arm in arms:
            row=allrows[arm,qid]
            r.update({arm+'_'+k:row.get(k) for k in ['candidate_recall','final_recall','recall_at_5','status','answer']})
            r[arm+'_grade']=grades[arm,qid]['accuracy'];r[arm+'_reason']=grades[arm,qid].get('reason','')
        review.append(r)
    with (root/'three-arm-review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(review[0]));writer.writeheader();writer.writerows(review)
    (root/'three-arm-review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
    pct=lambda v:f'{100*v:.2f}%'
    lines=['# 中文 20 题：四处修复后的三组对照','','日期：2026-11-12。正式修复组新生成 20 份答案及对应评分（另有两次中止试跑，详见调用量）；轻量版、修复前使用上一轮已冻结的 40 份结果。没有跑千题，也没有对旧答案选择性重评。','','## 结果','','| 指标 | 轻量版 | 新版修复前 | 新版修复后 |','| --- | ---: | ---: | ---: |']
    for group,title in [('overall','完整答对（全部）'),('2','完整答对（双文档）'),('3','完整答对（三文档）')]:
        cells=[]
        for arm in arms:
            row=summary[group][arm];n=row['questions'];correct=row['grades'].get('correct',0);cells.append(f'{correct}/{n}（{pct(correct/n)}）')
        lines.append('| '+title+' | '+' | '.join(cells)+' |')
    for key,title in [('candidate_recall','候选来源召回'),('recall_at_5','最终证据 Recall@5'),('final_recall','最终证据来源召回')]:
        lines.append('| '+title+' | '+' | '.join(pct(summary['overall'][arm][key]) for arm in arms)+' |')
    lines.append('| 最终找齐全部标准来源 | '+' | '.join(str(summary['overall'][arm]['all_sources'])+'/20' for arm in arms)+' |')
    lines+=['','来源召回按每题命中的标准来源比例计算，再对题目平均；不等于关键事实召回。Recall@5 使用最终证据列表的前 5 段。完整答对按原机器裁判的 correct 口径；运行状态 complete/partial 不用于替代正确率。','']
    for other,title in [('bilingualism','修复前'),('light','轻量版')]:
        p=pairs[other];lines+=[f"相对{title}：完整答对率差 {p['difference']*100:+.2f} 个百分点，救回 {len(p['improved'])} 题、退化 {len(p['regressed'])} 题。","",f"- 救回：{', '.join(p['improved']) or '无'}。",f"- 退化：{', '.join(p['regressed']) or '无'}。",f"- 20,000 次配对重采样 95% 区间：[{p['bootstrap_95_percentile'][0]*100:+.2f}, {p['bootstrap_95_percentile'][1]*100:+.2f}] 个百分点。",'']
    lines+=['## 控制变量与复算','','题目、全部 2,309 份语料、2,993 个片段及其向量/关系完全沿用上轮。快照：`'+manifest['new_snapshot']+'`。没有新抽取或更换 BGE-small。查询/回答模型仍为 Qwen3.5-35B-A3B，评分仍为 Qwen3.6-35B-A3B，温度 0、关闭思考，回答上限 2,000 token。阶段输出上限、候选池、上下文预算与最多两轮补查保持原配置；改变的是修复后的查询代码和提示协议。','','只复测修复组避免重复开销；两组旧基线与修复组调用时间不同，服务侧漂移无法完全排除。样本已用于定位缺陷，因此这是开发回归，不是独立盲测。已知裁判读漏问题沿用旧记录并明确保留，不能把 20 题成绩外推为千题总体准确率。','','复算检查：60 份答案对应完整的 20 个三组配对；来源召回逐题从原始来源 ID 重算，新旧新版引文偏移逐条核对；冻结快照内容校验通过，原基线文件未变化，运行期间修复版源码指纹未变化。','','## 修复行为诊断','','| 触发题数（不等于答错数） | 修复前 | 修复后 |','| --- | ---: | ---: |']
    for key,title in [('numeric_rule_triggered','数字规则触发'),('citation_rule_triggered','引用规则触发'),('selection_failed','选证技术失败'),('selection_fallback','保留未确认原文回退'),('verification_retrieval','核验缺证后定向补查'),('answer_repair','纯回答重写'),('answer_fallback','生成或核验异常时保留答案'),('sources_lost','候选齐全但最终缺来源')]:
        lines.append(f"| {title} | {diagnosis['bilingualism'].get(key,0)} | {diagnosis['fixed'].get(key,0)} |")
    lines+=['',f"最终引用块不匹配：修复前 {len(invalid['bilingualism'])} 题，修复后 {len(invalid['fixed'])} 题。数字规则触发可能是真实事实错误，也可能仍需人工诊断，不直接视作误报。",'', '## 调用量','','| 查询阶段（不含评分） | 轻量版 | 修复前 | 修复后 |','| --- | ---: | ---: | ---: |']
    for key,title in [('http_attempts','HTTP 尝试'),('prompt_tokens','输入 token'),('completion_tokens','输出 token'),('total_tokens','总 token')]:
        lines.append('| '+title+' | '+' | '.join(f'{costs[arm].get(key,0):,}' for arm in arms)+' |')
    lines+=['',f'中止试跑另产生 {pilot_generations} 份已落盘答案、{pilot_usage.get("http_attempts",0)} 次查询 HTTP 尝试、{pilot_usage.get("total_tokens",0):,} 个已返回计量的查询 token。两次中间试跑分别发现行内编号遗漏及补查后失败丢失旧答案的问题后停止，补齐离线回归后重新冻结源码；试跑不参与三组成绩，日志保留在 `.eval/crud-repair-pilot*-20261112/`。本次实际增量调用量应包含试跑、正式修复组及两者评分，不能只计算正式表格。', '', '用量只统计服务返回的计量，超时可能产生未返回用量；token 比不是账单费用比。查询耗时含共享限流和重试，不用于部署延迟结论。','','## Jev 与复现','','英文已有固定上游 Jev；中文本次补齐了关系评分和原文评分。真实服务尚无 TYPESAFE_API_KEY，已完成本地 HTTP 协议和错误回退测试。为遵守相同模型对照，本轮 `jev_enabled=false`，没有把 Jev 接入算成此次成绩来源。详见[中文链路修复与 Jev 接入](中文链路修复与Jev接入.md)。','','离线全套测试：65 通过、1 项 OCR 跳过，4 项子测试通过。','','脚本：`evaluation/retest_repairs.py`；离线报告：`evaluation/report_repairs.py`。本机材料位于 `.eval/crud-repair-20-20261112/`（Git 忽略）：`manifest.json`、`generations.jsonl`、`grades.jsonl`、`api-usage.jsonl`、`model-responses.jsonl`、`three-arm-summary.json`、`three-arm-review.csv`、`integrity-after.json`。原始基线目录不变。','']
    if (root/'case-review.md').exists():
        lines += [(root/'case-review.md').read_text()]
    args.report.write_text('\n'.join(lines))
    print(json.dumps({'overall':summary['overall'],'pairs':pairs,'diagnosis':diagnosis,'costs':costs},ensure_ascii=False))

if __name__=='__main__':main()
