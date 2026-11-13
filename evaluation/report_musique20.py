"""Offline MuSiQue report; reads frozen logs, never invokes a model."""
from pathlib import Path
import sys,json,hashlib,collections
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'.eval/musique20-20261113'
sys.path.insert(0,str(OUT/'official'))
from metrics.answer import compute_exact,compute_f1

def rows(name):
    p=OUT/name;return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
qs={q['id']:q for q in rows('questions.jsonl')};aa=rows('answers.jsonl');ss=rows('scores.jsonl')
assert len(aa)==len({x['id'] for x in aa});assert len(ss)==len({x['id'] for x in ss})
answers={a['id']:a for a in aa};scores={s['id']:s for s in ss}
assert set(answers)==set(scores)==set(qs) and len(qs)==20
records=[]
for qid,q in qs.items():
    a=answers[qid];s=scores[qid];pred=a['result']['answer']; golds=[q['answer']]+q['answer_aliases']
    records.append({'id':qid,'hops':q['hops'],'question':q['question'],'reference':q['answer'],'aliases':q['answer_aliases'],'prediction':pred,'raw_em':max(compute_exact(g,pred) for g in golds),'raw_f1':max(compute_f1(g,pred) for g in golds),'semantic_correct':s['correct'],'ambiguity':s.get('ambiguity',False),'judge_reason':s['reason'],'all_support':a['all_support'],'support_recall':a['support_recall'],'selected_count':len(a['selected_ids']),'gold_count':len(a['gold_ids']),'missing_support':sorted(set(a['gold_ids'])-set(a['selected_ids']))})
def aggregate(rs):
    n=len(rs)
    return {'n':n,'semantic_correct':sum(x['semantic_correct'] for x in rs),'semantic_accuracy':sum(x['semantic_correct'] for x in rs)/n,'raw_em':sum(x['raw_em'] for x in rs)/n,'raw_f1':sum(x['raw_f1'] for x in rs)/n,'all_support_count':sum(x['all_support'] for x in rs),'all_support_rate':sum(x['all_support'] for x in rs)/n,'support_recall':sum(x['support_recall'] for x in rs)/n,'judge_ambiguity_flags':sum(x['ambiguity'] for x in rs)}
usage=collections.defaultdict(collections.Counter)
for x in rows('llm.jsonl'):
    u=x.get('usage') or {};usage[x['role']].update({'calls':1,'prompt_tokens':u.get('prompt_tokens',0),'completion_tokens':u.get('completion_tokens',0),'usage_unknown':int(not u)})
jev=rows('jev-http.jsonl');errors=rows('errors.jsonl');triplets=rows('triplets.jsonl')
result={'offline_review':json.loads((OUT/'manual-review.json').read_text()),'overall':aggregate(records),'by_hops':{str(h):aggregate([x for x in records if x['hops']==h]) for h in (2,3,4)},'llm_usage':dict(usage),'jev':{'http_attempts':len(jev),'http_status':dict(collections.Counter(str(x.get('http_status','transport_error')) for x in jev)),'reported_charge_usd':str(sum((Decimal(x['charge_usd']) for x in jev if x.get('charge_usd') is not None),Decimal(0))),'unknown_charge_attempts':sum(x.get('charge_usd') is None for x in jev),'charge_source':'JevHub response header; not verified against invoice','usage_unknown_attempts':sum(not x.get('usage') for x in jev),'input_tokens':sum((x.get('usage') or {}).get('input_tokens',0) for x in jev) if any(x.get('usage') for x in jev) else None,'output_tokens':sum((x.get('usage') or {}).get('output_tokens',0) for x in jev) if any(x.get('usage') for x in jev) else None},'operational_restarts':rows('operational-restarts.jsonl'),'llm_errors_by_stage':dict(collections.Counter(x['role'] for x in errors)),'official_dataset_verification':json.loads((OUT/'official-dataset-verification.json').read_text()),'llm_errors':len(errors),'triplets':{'paragraphs':len(triplets),'empty':sum(x['empty'] for x in triplets),'native_parse_fallbacks':sum(x.get('extraction_failed',False) for x in triplets),'count':sum(len(x['triplets']) for x in triplets)},'records':records}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# MuSiQue 英文 20 题评测（2026-11-13）','', '## 口径','', '固定随机种子 20261113，从 MuSiQue-Ans v1.0 开发集抽取二跳 8 题、三跳 6 题、四跳 6 题。保留每题所有候选段落和干扰段落，386 段去重后建一个共享索引；查询使用上游原生 passage ID 过滤。初始实体/关系检索仍在共享索引上进行，再执行上游过滤；因此这不是每题独立 20 段索引，也不是官方排行榜配置。原始段落作为 passage，不再次切分。标准答案、问题分解和支持标签均未进入建库、检索或回答。','', '英文链路：固定 Zilliz 上游 commit `47590ad4eab5838cdcfec94f8a53ebfe70ebe330`；本地 BGE-large-en-v1.5 ONNX / 1024 维、Milvus Lite，原生图扩展、Jev 1.13.0 关系和段落两阶段评分、final_top_k=10。抽取和回答使用 Qwen/Qwen3.5-35B-A3B；语义评分使用 Qwen/Qwen3.6-35B-A3B。保留上游提示词；调用适配仅关闭 thinking、设置输出上限、检查完成状态和记录 token。','', '严格 EM/F1 使用随归档保存的 MuSiQue 官方评分源码，直接评价未经改写的上游答案，支持答案别名。上游常输出完整句子，因此严格 EM 不能单独反映事实正确率。参考答案语义匹配率是另一个模型的判断，允许同义实体名和完整句式；不等同于官方 EM。支持段落召回只表示找到对应段落，不保证每一步推理都正确。','', '## 结果','', '|分组|题数|参考答案语义匹配|原始 EM|原始 F1|支持段落找齐|平均支持召回|','|---|---:|---:|---:|---:|---:|---:|']
for name,m in [('全部',result['overall'])]+[(f'{h} 跳',result['by_hops'][str(h)]) for h in (2,3,4)]:
    lines.append(f"|{name}|{m['n']}|{m['semantic_correct']}/{m['n']}（{m['semantic_accuracy']:.1%}）|{m['raw_em']:.1%}|{m['raw_f1']:.1%}|{m['all_support_count']}/{m['n']}|{m['support_recall']:.1%}|")
lines+=['','## 费用与异常','',f"Jev 共 {len(jev)} 次 HTTP 请求，响应头累计报告 ${result['jev']['reported_charge_usd']}；{result['jev']['unknown_charge_attempts']} 次未报告费用。这是网关响应记录，不是账单核验。SiliconFlow 未提供逐次金额，以下保留原始 token 用量，不将未知费用记成零。",'', '|调用阶段|收到响应数|输入 tokens|输出 tokens|','|---|---:|---:|---:|']
for role,u in usage.items():lines.append(f"|{role}|{u['calls']}|{u['prompt_tokens']}|{u['completion_tokens']}|")
lines+=['',f"LLM 记录异常 {len(errors)} 次；Jev 状态分布：{result['jev']['http_status']}。{len(triplets)} 段抽取出 {result['triplets']['count']} 条关系，空抽取 {result['triplets']['empty']} 段。没有静默切换其他模型或中文链路。",'', '运行中为调整超时、并发和流式接收重启过两次，最多 9 个在途 SiliconFlow 请求以及超时尝试的完整用量/计费未知；已完成的段落从落盘记录恢复。早期抽取输出上限为 4096，之后为 2048；触及上限的响应被拒绝并记录重试，没有作为有效关系用于建库。多次失败且留下不完整 JSON 的段落，由 `evaluation/recover_musique_native_json.py` 离线重放上游 `_parse_response`，得到空关系列表，并记录 extraction_failed；原文及其向量照常入库。这是明确记录的图信息缺失，不是抽取成功。第二次恢复改用流式接收，仍收齐完整响应后才写入结果。', '']
review=result['offline_review']
lines+=['## 离线复核：分数含义与数据问题','', '20 道题均逐题检查了最终回答与参考答案的一致性，保持原评分标签：10/20（50%）。这是参考答案语义匹配，不是对长回答每条解释均正确的保证。没有剔除问题题目、换样本或重跑答案。','', f"离线发现 {review['data_or_wording_issue_count']} 题存在题干、标注衔接或支持事实问题；其中 {sum(not x['reference_match'] for x in review['issues'])} 题未匹配参考答案。它们并非全都等于参考答案本身错误，也不能据此把所有未匹配都归为模型错误或全部免罚。",'', '|题号 / ID|问题类型|证据与处理|','|---|---|---|']
for x in review['issues']:lines.append(f"|{x['ordinal']} / {x['id']}|{x['type']}|{x['note']}|")
lines+=['', '较明确的链路问题：','']
for x in review['clear_chain_failures']:lines.append(f"- {x['id']}：{x['note']}")
lines+=['', review['extra_rationale_caveat']['id']+'：'+review['extra_rationale_caveat']['note'], '', '评分器原始理由也有缺陷，日志完整保留，不用于证明世界事实：', '']
for note in review['judge_reason_caveats']:lines.append('- '+note)
lines+=['', '因此，50% 只表示这套配置在这 20 道固定题上的参考答案匹配率；不宜直接与官方短答 EM/F1 或更大样本、其他配置比较。标注支持段落找齐，也不保证其中真的写明了全部关键事实。','']
lines+=['## 逐题结果','']
for x in records:
    lines += [f"### {x['id']}（{'匹配参考答案' if x['semantic_correct'] else '未匹配参考答案'}）",'',x['question'],'',f"参考答案：{x['reference']}；别名：{json.dumps(x['aliases'],ensure_ascii=False)}。",'',f"模型回答：{x['prediction']}",'',f"支持段落：{x['gold_count']-len(x['missing_support'])}/{x['gold_count']}；评分器原始理由（不能直接作为事实依据）：{x['judge_reason']}；歧义标记：{x['ambiguity']}。",'']
lines+=['## 复核与复跑','', '`evaluation/run_musique20.py` 保存题目、原文、抽取结果、不可变快照、逐题答案、评分和计费日志。`evaluation/report_musique20.py` 仅离线重算统计，不调用模型。原始材料位于 `.eval/musique20-20261113/`。本机复用 `/tmp/oa-chroma-models` 中的英文模型缓存；归档不包含模型权重，版本信息见 embedding-assets.json。','', '来源：[MuSiQue 官方项目](https://github.com/StonyBrookNLP/musique)，开发集已与官方仓库 download_data.sh 指定的原始压缩包逐字节核对，完全一致；证据见 official-dataset-verification.json。下载镜像及 SHA-256 见 manifest.json。官方评分源码及许可证保存在原始材料的 official/ 目录。','', '样本只有 20 题，每题改变总正确率 5 个百分点，适合检查链路效果，不能视为全量准确率。']
(ROOT/'docs/MuSiQue英文20题评测.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))
