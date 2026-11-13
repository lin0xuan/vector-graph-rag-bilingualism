# Jev 官方与 JevHub 双服务配置

官方是默认服务；JevHub 是可选的第三方网关。两者复用固定 Zilliz 上游的 System One 请求、评分验证与重排逻辑，不修改候选排序算法。提供网关配置不代表项目验证或背书其模型来源、可用性或账单。

## 配置选择

中英文 CLI 都支持顶层 `reranker`。中文同时用于关系和原文评分；英文仍由原生上游完成关系/原文重排，仅将连接设置传给上游。

```json
{"reranker": {"provider": "jev", "service": "typesafe", "model": "jev-1.13.0"}}
```

```json
{"reranker": {"provider": "jev", "service": "jevhub", "model": "jev-1.13.0"}}
```

| service | 默认 base_url | 默认 key_env | 默认 max_concurrency |
|---|---|---|---|
| typesafe | https://api.typesafe.ai/v1 | TYPESAFE_API_KEY | 3 |
| jevhub | https://api.jevgo.cn/v1 | JEVHUB_API_KEY | 2 |

`base_url`、`key_env`、`timeout`、`max_concurrency` 可显式配置。地址包含 `/v1`，客户端追加 `/systemone`。不按密钥猜服务、不跨服务自动使用另一把密钥，也不在失败时切换供应商。保留原 `provider=jev` 配置及显式自定义地址的兼容性。

英文原 `examples/en.json` 保留原生默认行为；中文官方样例为 `examples/zh-jev.json`。第三方样例为 `examples/en-jevhub.json` 和 `examples/zh-jevhub.json`。样例里的回答/嵌入服务需要自行填写。切换已有中文知识库时，只修改重排配置，保持原向量模型、分词器和索引。英文当前将服务配置纳入不可变索引清单，不能直接更换服务后套用旧快照；应使用匹配配置和独立集合，切勿覆盖旧集合。

本地 `.env` 分别填写：

```dotenv
TYPESAFE_API_KEY=
JEVHUB_API_KEY=
```

不使用哪家就留空。当前英文档案固定 `jev-1.13.0`；中文也应固定版本，拒绝 `jev-latest`、`jev-preview`，避免对照过程中模型变化。

## 费用、用量与错误

- 官方响应：记录 `usage.input_tokens/output_tokens`，不在程序中硬编码会变动的价格。
- JevHub 响应：允许缺少 `usage`；记录 `X-JevHub-Charge-USD`，用 Decimal 累加服务报告的美元扣费。缺少金额或用量标为未知，不能据此断言免费。
- 所有 HTTP 尝试（包括重试、错误状态和成功但评分结构无效的响应）均统计。观察器仅接收状态码、请求哈希、模型、评分阶段、用量、金额和耗时，不含密钥或原文。
- Python 中文适配器支持 `request_observer=callback`，用于逐请求 JSONL 落盘；累计统计通过 `adapter.accounting.snapshot()` 读取。英文显式选择服务后，通过 `pipeline.jev_accounting.snapshot()` 读取累计统计。这些统计在内存中，长期留存需由调用方持久化。
- 网关原始金额不是项目独立核实的账单。以服务返回的实际模型字段记账，不把返回别名当成独立验证的模型版本。
- 网关单个客户端限制每秒最多约 4 次起始请求、最多配置 4 并发，默认 2。多个进程/客户端共享同一密钥时仍需外部统一限流；服务的 429 按原上游退避策略重试。
- 原上游重试保留：临时网络错误及 429/500/502/503/504/529 最多 3 次；401/402/403 不重试。中文服务失败在查询 trace 中保留回退标记；英文保持上游异常传播行为。

JevHub 协议依据为用户提供的 `jevhub-api.md`，不是 Typesafe 官方承诺。官方文档：https://docs.typesafe.ai/api/system-one 。

## 本地验证

离线测试用本地 HTTP 服务分别模拟官方 usage 和网关无 usage + 扣费头，覆盖中文两阶段、英文原生重排、超过 16 个候选、鉴权失败不重试、付费成功响应但评分无效仍记账、独立密钥和清单不含凭证。测试禁止访问远程模型，不能替代真实连通性或准确率评测。
