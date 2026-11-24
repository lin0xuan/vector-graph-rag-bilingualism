# 中文链路修复与 Jev 接入

日期：2026-11-12。查询版本：`coverage-v2`。索引格式不变，旧不可变快照可以直接复用。

## 四处修复

1. **选证丢证据**：选证输入使用局部 `p1` 等片段编号和原文句子编号，输出仅返回编号；程序恢复原文及字符偏移。提示要求保留同一问点的互补事实。模型超时、输出截断、无效编号时保留相关批次的候选原文，在上下文预算内提供未确认材料，记录 `selection_fallback`；这些材料不会伪造引文，也不会直接计为问点已覆盖。正常返回空证据仍表示没有选中支持。
2. **数字核验误报**：先去除引用、Markdown 格式、行首列表序号、行内连续编号与明确的问点编号，再核对事实数字。统一千位分隔和小数末尾零等值格式。保留金额、日期、百分比等实际数字检查；不能把 1.5 万元当成列表 1 而删除。该规则仍是字面数字检查，不是数学或语义证明。
3. **引用 ID 抄错**：模型只看短片段 ID，程序把最终引用精确映射到原稳定 ID。支持 `[p1,p2]` 规范化为两个引用，不对不存在的编号做模糊猜测。补查改变上下文后重建映射并重新回答，避免旧编号指向新片段。完整映射保存在 trace 中。
4. **缺证后不补查**：核验器区分“证据已在上下文、答案漏写”和“原文确实缺证”。前者重写答案，后者返回 `gaps` 中的问点和具体检索问题。缺证触发定向召回、重新选证和重新回答；可以重新检查先前进入候选池却没被选上的片段。初始覆盖补查与核验补查共用最多两轮预算，另外最多一次纯回答重写，防止无界循环。补查后的回答生成失败时，恢复上一份可用答案及与其匹配的原文/引文，保留缺证标记，不把旧答案配上新上下文。每次生成的可用答案写入 trace，答案和核验的原始响应另存于评测忽略目录，便于定位中间失误。

运行状态 `complete` 表示工程流程和本轮核验通过，不等于外部评测答对。保留过技术回退的结果仍标记 `partial`，最终答案与证据可以继续用于诊断。

## 中文 Jev 的真实位置

英文从首版起就通过固定版本的 Zilliz 上游调用 Jev；首版中文没有 Jev，`two-stage` 只是“生成式关系筛选 + 原文选证”，不能将这个模式名理解成已经使用 Jev。

现在中文新增 `src/vector_graph_rag_bilingualism/jev.py`，复用固定上游提交 `1c034849fb36a104dadb1ddf967abcfe5bbd99b7` 的 `JevReranker`，通过真实 `/systemone` 协议读取 `noul` 分数：

- 关系阶段：用问题和候选关系共享上下文给关系评分，排序后参与图通道融合；替代生成式关系筛选。
- 原文阶段：给融合后的候选原文评分，再进行中文问点选证和回答。每批最多 16 段，保留全部候选，不把 100 个候选静默截为 16 个。
- 两阶段都验证响应，服务失败写入回退 trace，保留原候选。配置缺少密钥则在连接阶段明确报错，不偷偷宣称使用 Jev。
- 中文保留自己的候选池和问点覆盖策略。因此这不是逐行复制英文的“最多 16 个合并候选”配方；跨批共享上下文不同，中文效果要独立评估。
- `single-stage` 跳过关系评分，仍可使用 Jev 原文评分；中文 Jev 双阶段配置应使用 `two-stage`（默认）。

上游代码依据：[Jev 两阶段实现](https://github.com/zilliztech/vector-graph-rag/blob/1c034849fb36a104dadb1ddf967abcfe5bbd99b7/src/vector_graph_rag/llm/jev.py)、[RAG 调用位置](https://github.com/zilliztech/vector-graph-rag/blob/1c034849fb36a104dadb1ddf967abcfe5bbd99b7/src/vector_graph_rag/rag.py)。

## 启用

```bash
uv sync --locked --extra jev --extra api --extra documents --extra test
```

本机 `.env` 中填写 `TYPESAFE_API_KEY`，生成和嵌入模型仍使用各自的环境变量。配置样例是 `examples/zh-jev.json`，需要将其中的生成/嵌入地址改为自己的有效服务。已有知识库应在其原配置上只增加 `reranker`，保留原 embedding、namespace、data_dir 和 tokenizer，不能把 BGE-small 索引切到 BGE-M3 配置后继续查询。

```json
{
  "reranker": {
    "provider": "jev",
    "model": "jev-1.13.0",
    "key_env": "TYPESAFE_API_KEY",
    "base_url": "https://api.typesafe.ai/v1",
    "timeout": 60,
    "max_concurrency": 3
  }
}
```

```bash
uv run vgrag-bi --config examples/zh-jev.json --env-file .env verify-index
uv run vgrag-bi --config examples/zh-jev.json --env-file .env query '你的问题'
```

不提供 `reranker` 的旧中文配置继续使用固定生成模型，避免在没有 Jev 凭证时暗中切换服务。模型、服务端点和 Jev 固定版本均纳入查询配置指纹；凭证值不进入清单或日志。

## 验收与本轮对照边界

完整离线测试：65 通过、1 项真实 OCR 因缺少 Tesseract 跳过，另有 4 项子测试通过。覆盖短 ID/原文偏移、编号误报、真实数字错误、选证失败保留原文、核验后重新检查已见候选、共享补查预算；中文 Jev 使用真实上游客户端、本地 HTTP 返回受控响应，验证两个阶段、20 段跨批排序、重复文本不丢 ID、坏响应和可见回退。

本机检查时没有配置 `TYPESAFE_API_KEY`，因此尚未对真实 Jev 服务进行中文调用或准确率评测。离线接口验收不能证明中文质量，也不等于已获得服务授权。

用户要求的这轮 20 题复测固定原 Qwen 模型、BGE-small 冻结快照、题目、回答输出上限和评分提示，因此明确 **不启用 Jev**。仅重新生成修复版的 20 份答案，轻量版和修复前两组使用已保存的原始结果；没有重建关系或运行千题。


## 密钥申请、收费与下一轮 20 题对照

核对日期：2026-11-12。进入 [Typesafe 官方密钥页面](https://console.typesafe.ai/settings/keys)，使用 Google 或邮箱登录，在控制台创建 API Key；在本项目 `.env` 的 `TYPESAFE_API_KEY=` 后填写。充值入口为控制台 Settings → Billing。

[官方计费说明](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost)标价为每百万输入 token 0.042 美元，输出免费；预充值扣费，不提供免费试用额度。例如实际计费输入为 1,000 万 token 时，Jev 费用为 0.42 美元。这个例子不是 20 题的费用预测，也不包含 Qwen 回答和评分费用。最终以服务返回的用量及账单为准。

以下是最初记录的对照约束。后续已增加官方/JevHub 双服务支持，并通过 JevHub 完成原 20 题复测（14/20 完整答对）；详见[双服务配置](Jev双服务配置.md)与[启用前后对照](中文20题Jev启用前后对照.md)。

- 使用 `.eval/crud-ab-20-20261112/questions.jsonl` 原 20 题（双文档 10、三文档 10），保留全部 2,309 份语料。
- 复用快照 `8ad452a6d82a6d57a704a3331f7e9cbaa634adec3f783f225c42fb96e56d4c7c`，验证索引内容哈希；不重建关系、不替换 BGE-small 向量。
- 未启用 Jev 的基线使用 `.eval/crud-repair-20-20261112` 正式修复结果（11/20 完整答对），不是修复前的 10/20；旧轻量版 18/20 仅作参考。
- 固定 Qwen/Qwen3.5-35B-A3B 回答模型、Qwen/Qwen3.6-35B-A3B 评分模型、提示词、温度、输出上限和检索预算。只将关系与段落重排切换到 Jev `jev-1.13.0`，其余中文选证/核验保留。
- 单独输出 Jev 组结果，保存服务错误与回退情况，比较证据召回、完整答对率、逐题改善/退化、调用用量和延迟。认证/余额问题先停止处理，不能将全程回退当成成功启用 Jev。
- 本轮原 `evaluation/retest_repairs.py` 明确禁用 Jev，不能直接当作 Jev 评测入口；后续已添加独立入口 `evaluation/retest_jev.py` 及配置清单，未覆盖历史对照。
- 不扩题、不运行全量千题；填好后由用户告知再开始，不轮询密钥或后台自动启动。
