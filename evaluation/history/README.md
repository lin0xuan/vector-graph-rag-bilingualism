# 轻量版历史实验（只读档案）

这些是 chinese-light 已公开的精简记录与标准库复算脚本，保留旧题目/片段 ID、分数和复核分类。
它们不是 bilingualism 新链路的运行结果，不包含完整语料、原始回答或密钥。

```bash
python evaluation/history/replay.py
```

- `chinese-1000.jsonl`：历史千题，791/1000 完整答对。
- `rerank-ab-322.jsonl`：历史固定候选重排 A/B。
- `fact-audit-43.jsonl`：历史事实复核。
- `answer-ab-32.jsonl`：历史回答提示词比较。

原始日志仍位于旧项目 `.vgrag-eval/historical-oa/`；需要时给 replay.py 显式传入 `--raw-dir`。
原千题完整候选次序与 322 题两臂回答全文存在历史归档缺口，迁移并未凭空补齐。
500 题的最新离线复盘也保留在旧项目 `.vgrag-eval/crud-500.lqeu5e/review/`，未复制大批原文到新仓库。

来源提交：`lin0xuan/vector-graph-rag-chinese-light@1ba62ecfc7a15af3d412f7039a68b389080fad4f`。
新实验的生成、评分、定向补跑接口见 `vector_graph_rag_bilingualism.evaluation.EvaluationRunner`。
