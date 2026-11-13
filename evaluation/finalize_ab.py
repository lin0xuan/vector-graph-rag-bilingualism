"""Verify completed A/B records and render a reproducible report, without model calls."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,re,sqlite3,statistics
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args(); root=args.directory
    manifest=json.loads((root/'manifest.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    paired=json.loads((root/'paired-statistics.json').read_text())
    diagnosis=json.loads((root/'runtime-diagnosis.json').read_text())
    questions={r['id']:r for r in read(root/'questions.jsonl')}
    answers={(r['arm'],r['id']):r for r in read(root/'generations.jsonl')}
    grades={(r['arm'],r['id']):r['grade'] for r in read(root/'grades.jsonl') if r.get('grade')}
    expected={(arm,qid) for arm in ('light','bilingualism') for qid in questions}
    assert len(questions)==20 and set(answers)==expected and set(grades)==expected
    with sqlite3.connect(root/'light-index/index.sqlite3') as db:
        id_source=dict(db.execute('SELECT id,source FROM passages'))
    checked=[]
    for key,row in answers.items():
        gold=set(questions[row['id']]['sources'])
        final={p['source'] for p in row['citations']}
        top5={p['source'] for p in row['citations'][:5]}
        assert abs(len(final&gold)/len(gold)-row['final_recall'])<1e-12
        assert abs(len(top5&gold)/len(gold)-row['recall_at_5'])<1e-12
        if row['arm']=='light':
            candidate={p['source'] for p in row.get('candidates',[])}
        else:
            ids=set()
            for event in row.get('trace',{}).get('events',[]):
                if event['stage']=='fusion': ids.update(event['ids'])
                elif event['stage']=='repair': ids.update(event['added'])
            candidate={id_source[pid] for pid in ids}
        assert abs(len(candidate&gold)/len(gold)-row['candidate_recall'])<1e-12
        if not (row.get('answer') or '').strip(): assert grades[key]['accuracy']=='wrong'
        checked.append(key)
    for arm in ('light','bilingualism'):
        assert sum(g['accuracy']=='correct' for (a,_),g in grades.items() if a==arm)==summary['overall'][arm]['complete_answers']
    assert json.loads((root/'integrity-after.json').read_text())['verified']
    events=read(root/'api-usage.jsonl')
    usage=defaultdict(Counter)
    for event in events:
        label='judge' if event['role']=='judge' else 'query'
        counts=usage[(event['arm'],label)]
        # Non-stop replies have both a completed HTTP event and a logical-failure event.
        # Count that physical request once, while keeping its measured token usage.
        if event['status']=='complete' or 'finish_reason' not in event:
            counts['http_attempts']+=1
            counts['http_elapsed_ms']+=event.get('elapsed_ms',0)
        if event['status']=='complete':
            for k in ('prompt_tokens','completion_tokens','total_tokens'):
                counts[k]+=(event.get('usage') or {}).get(k,0)
        if event['status']=='failed': counts['failure_events']+=1
    query={arm:dict(usage[arm,'query']) for arm in ('light','bilingualism')}
    citation_errors=[]
    for (arm,qid),row in answers.items():
        if arm!='bilingualism':continue
        refs=re.findall(r'\[([^\[\]]+)\]',row.get('answer') or '')
        allowed={p['id'] for p in row['citations']}
        invalid=sorted(set(refs)-allowed)
        if invalid: citation_errors.append({'id':qid,'invalid_refs':invalid})
    audit={'verified_question_pairs':20,'verified_answers':len(checked),'same_corpus_integrity_verified':True,
           'query_usage':query,'judge_usage':{arm:dict(usage[arm,'judge']) for arm in ('light','bilingualism')},
           'final_invalid_citations':citation_errors,
           'metrics_note':'Recall is macro averaged over question source IDs; not fact-level recall.'}
    (root/'independent-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    pct=lambda value:f'{100*value:.2f}%'
    lines=['# 中文 20 题真实模型 A/B：轻量版与 bilingualism 首版','',
           '日期：2026-11-12。范围：20 道题，两版共 40 份现生成答案；没有运行全量千题或重建全量图。','',
           '## 结果','', '| 指标 | 轻量版 | 新版 | 新版减轻量版 |','| --- | ---: | ---: | ---: |']
    for group,title in [('overall','完整答对（全部）'),('2','完整答对（双文档）'),('3','完整答对（三文档）')]:
        a,b=summary[group]['light'],summary[group]['bilingualism']
        x,y=a['complete_answers']/a['questions'],b['complete_answers']/b['questions']
        lines.append(f"| {title} | {a['complete_answers']}/{a['questions']}（{pct(x)}） | {b['complete_answers']}/{b['questions']}（{pct(y)}） | {(y-x)*100:+.2f} 个百分点 |")
    a,b=summary['overall']['light'],summary['overall']['bilingualism']
    for key,title in [('candidate_recall','候选来源召回'),('recall_at_5','Recall@5'),('final_recall','最终来源召回')]:
        lines.append(f"| {title} | {pct(a[key])} | {pct(b[key])} | {(b[key]-a[key])*100:+.2f} 个百分点 |")
    for key,title in [('candidate_all_sources','候选找齐全部标准来源'),('final_all_sources','最终找齐全部标准来源')]:
        lines.append(f"| {title} | {a[key]}/20 | {b[key]}/20 | {b[key]-a[key]:+d} 题 |")
    lines += ['',f"配对结果：新版救回 {paired['improved']} 题、退化 {paired['regressed']} 题。改善题：{', '.join(summary['paired']['improved']) or '无'}。退化题：{', '.join(summary['paired']['regressed']) or '无'}。",'',
              '这些是本次配置的端到端结果，包含响应截断、解析失败和服务拒答；没有将失败题悄悄剔除。`complete/partial` 运行状态与外部判分是两套口径。', '',
              '召回口径：每题按命中的标准来源数 / 标准来源总数计算，再对题目取平均；不是关键事实召回。候选为本轮及补查候选的并集，Recall@5 为最终证据列表前 5 段的来源召回，最终召回使用完整最终证据列表。来源找齐不保证关键事实齐全。', '',
              '## 控制变量和未覆盖范围','',
              '- 从原千题中按固定种子 `2026111202` 抽双文档、三文档各 10 题；排除之前 500 题实验及切分 A/B 已用题目。它仍是探索性开发样本，不是全新独立基准。',
              '- 两版均使用全部 2,309 份语料、2,993 个片段，包含干扰材料；沿用 500 题实验的冻结 BGE-small 中文向量和历史关系。索引运行前后均校验，未重建关系。',
              '- 轻量版为提交 `1ba62ecfc7a15af3d412f7039a68b389080fad4f` 的当前查询代码；新版源码文件 hash 全部记录在 manifest。两版均现跑，没有用历史 79.1% 充当本轮基线。',
              f'- 所有查询模型固定为 `{CHAT}`，回答上限统一为 2,000 token、temperature=0、enable_thinking=false。评分使用 `{JUDGE}`，同一提示和同一证据字段形状，不提供系统名称给裁判。',
              '- 轻量版使用 single-stage、候选最多 24 段、最终最多 8 段；新版使用默认 two-stage、候选池 100、每通道 30、最多 6 个问点、最多 2 轮补查、证据正文预算 8,000 个 BGE token。',
              '- 新版仍用冻结 BGE-small 查询向量；未评测 BGE-M3、新关系抽取或 cross-encoder。轻量版 dense 为 Chroma HNSW，新版为同一组向量的精确余弦搜索。此实验对比查询链路整体，不是单模块消融。',
              '- 本轮模型适配器输出上限：plan 1,536、entities 256、relations 2,048、select 4,096、answer 2,000、verify 2,048、judge 1,000。故这里的成绩不能外推到无输出上限或已修复的未来版本。','',
              '## 调用开销','', '| 查询阶段（不含评分） | 轻量版 | 新版 |','| --- | ---: | ---: |']
    for key,title in [('http_attempts','HTTP 请求尝试'),('prompt_tokens','已返回计量的输入 token'),('completion_tokens','已返回计量的输出 token'),('total_tokens','已返回计量的总 token')]:
        lines.append(f"| {title} | {query['light'].get(key,0):,} | {query['bilingualism'].get(key,0):,} |")
    ratio=query['bilingualism']['total_tokens']/max(query['light']['total_tokens'],1)
    lines += ['',f"新版记录到的查询 token 为轻量版的 **{ratio:.2f} 倍**。失败请求可能没有返回完整用量；token 比不是精确账单费用比。",
              f"本轮平均查询墙钟时间为轻量版 {a['mean_query_ms']/1000:.1f} 秒、新版 {b['mean_query_ms']/1000:.1f} 秒，包含并发 2 下的共享限流等待与重试，不能作为独占部署延迟结论。",'',
              '## 已核对的问题','',
              '1. **数字核验误报。** `2docs:359` 的列表编号 1、2、3 被当成事实数字。已用无需模型的构造例复现。该题原文中有“车辆安全、职业规范、服务质量”，新版答案却漏答，核验主要在处理编号而未补齐事实。',
              '2. **长引用 ID 抄错。** 同一题修复前引用 ID 正确，修复后少抄了一个字符，新增引用错误。应让模型使用短编号，由程序映射回稳定 ID。',
              '3. **选证截断后丢掉可用候选。** `2docs:583` 的候选已找齐标准来源，但两次选证输出达到上限，最终没有可用证据，返回证据不足。应压缩结构化选证输出，并将技术失败与正常无证据分开处理。',
              '4. **核验发现缺口后仍只重写回答。** `3docs:312` 的核验指出问点缺证，但当前实现未把核验阶段的缺口送回定向检索。已有回答修复不能补回被选证丢掉的事实。',
              '5. **评分模型不能无条件视为真值。** `2docs:576` 轻量版被服务返回 `risky`，实际没有答案，裁判却误判为 correct。原始判分保留，统一“空答案按未答对”规则已纠正；该规则对两版都适用。',
              '6. **判分理由还存在读漏。** `3docs:310` 的两份答案都写了接送卡，裁判理由却声称未覆盖；新版也明确写了自信、自立、自强。两份答案仍有参考事实遗漏，但原始 wrong 等级的理由并不可靠。本轮保留原始判分，不做有利于某一版的选择性重评，因此表中完整答对率是机器裁判口径。',
              '7. **来源不等于事实。** 已核对 `3docs:266`、`3docs:162`、`2docs:3`、`3docs:534`：虽然少于全部标准来源，现有文本仍能回答核心问点。没有把这类来源遗漏直接改判为答案错误。','',
              f"运行诊断计数（触发不等于必然答错）：`{json.dumps(diagnosis['new_pipeline_observations'],ensure_ascii=False)}`。最终答案中引用块未被现有解析器识别的题数：{len(citation_errors)}（一题 ID 少字符，另一题把两个合法 ID 放在同一个方括号内）。",'',
              '## 小样本解释与后续优先级','',
              f"完整答对率配对差为 {paired['difference']*100:+.2f} 个百分点；20,000 次配对重采样区间为 [{paired['bootstrap_95_percentile'][0]*100:+.2f}, {paired['bootstrap_95_percentile'][1]*100:+.2f}] 个百分点。该区间不包含裁判误差，也不证明原千题总体的绝对变化。",
              '当前首版不宜直接替代轻量版。先修复短证据编号/句子编号映射、格式序号误报、选证失败回退与缺口反馈，再做同样受控的对照；当前候选已较充分，暂不应把更换更大嵌入模型当作主要解法。', '',
              '## 复现与材料','',
              '脚本：`evaluation/compare_light.py`、`evaluation/report_paired.py`、`evaluation/diagnose_ab.py`、`evaluation/finalize_ab.py`。本机完整材料位于 `.eval/crud-ab-20-20261112/`（Git 忽略目录）：', '',
              '- `manifest.json`、`protocol.json`：配置、源码指纹、语料和样本身份。',
              '- `questions.jsonl`、`generations.jsonl`、`grades.jsonl`：固定题目、逐题输出、全部评分记录及规则纠正。',
              '- `summary.json`、`paired-statistics.json`、`paired-review.csv`：总体/分组统计与配对对比。',
              '- `runtime-diagnosis.json`、`independent-audit.json`：阶段问题及独立复算校验。',
              '- `api-usage.jsonl`、`integrity-after.json`：实际调用用量、运行后索引完整性。', '',
              '原始数据、密钥和完整日志未加入公开文档；本轮没有运行全量千题。','']
    args.report.write_text('\n'.join(lines))
    print(json.dumps({'answers':len(checked),'query_token_ratio':ratio,'invalid_final_citations':len(citation_errors)},ensure_ascii=False))


CHAT='Qwen/Qwen3.5-35B-A3B'
JUDGE='Qwen/Qwen3.6-35B-A3B'
if __name__=='__main__': main()
