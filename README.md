# vector-graph-rag-bilingualism

准确率优先的中英文双链路 RAG。英文封装固定版本的 Zilliz 上游；中文围绕问点覆盖、可定位证据、定向补查与回答核验重新编排。

当前是可运行的首版：Python API、CLI、本地 HTTP 入口、不可变索引、英文接入、中文单/两阶段查询及断点恢复已经实现。中文四处修复后的 20 题真实模型对照：完整答对为轻量版 18/20、修复前 10/20、修复后 11/20；最终来源召回从 59.17% 提高到 84.17%，但回答与核验仍有明显问题。当前不宜据此替代轻量版。见[修复后对照](docs/中文20题修复后对照.md)，原始[首轮对照](docs/中文20题真实模型A-B.md)保留不变。中文 Jev 已支持官方与 JevHub 两套服务；通过 JevHub 固定 jev-1.13.0 的原 20 题复测为 14/20，最终来源召回 90.83%，三文档完整答对仍为 6/10。见[Jev 启用前后对照](docs/中文20题Jev启用前后对照.md)。

## 安装

Python 3.11+，使用 `uv.lock` 固定依赖。英文依赖固定到上游提交 `47590ad4eab5838cdcfec94f8a53ebfe70ebe330`。

```bash
uv sync --locked --extra english --extra api --extra documents --extra test
uv run vgrag-bi --help
```

只开发中文核心时可省略 `english`。中文默认使用 SQLite 中的精确向量检索进行开发；配置 `milvus` 后使用 Milvus 查询，并校验镜像实际内容。英文存储由上游 Milvus 负责。

## 配置与使用

```bash
cp .env.example .env
```

在 `.env` 填写自己的密钥，并修改 `examples/zh.json` 或复制为本地配置文件，设置模型地址、名称、向量维度。示例地址是本机服务占位值，不是已经运行的服务。不同角色可在 `models` 中分别配置 `extract / plan / entities / relations / select / answer / verify`，未设置的角色继承 `default`。

以下索引和问答命令会访问你配置的模型服务，测试命令则不会：

```bash
uv run vgrag-bi --config examples/zh.json --env-file .env index --documents examples/documents-zh.jsonl
uv run vgrag-bi --config examples/zh.json --env-file .env query '谁负责示例项目的审批，办理期限多久？'
uv run vgrag-bi --config examples/zh.json --env-file .env verify-index
```

中文 `query.retrieval_mode` 支持 `single-stage` 和 `two-stage`，切换只影响查询策略，可以共用同一索引。`--snapshot ID` 固定已有快照。更新语料时重新输入该知识库完整文档清单，构建新快照后再原子切换；已完成的抽取/向量任务默认复用持久化检查点。源文件删除通过新清单排除该来源，旧快照仍可追溯。

可选中文 Milvus 配置：

```json
{"milvus": {"uri": "http://127.0.0.1:19530", "token_env": "VGRAG_MILVUS_TOKEN"}}
```

中文已支持 Jev 关系评分和原文评分，使用固定上游 `/systemone` 协议；见 `examples/zh-jev.json` 和[中文修复与 Jev 接入](docs/中文链路修复与Jev接入.md)。需要 `uv sync --extra jev`。默认官方服务使用 `TYPESAFE_API_KEY`；JevHub 使用独立 `JEVHUB_API_KEY`，中英文分别提供 `examples/zh-jevhub.json`、`examples/en-jevhub.json`。详见[双服务配置](docs/Jev双服务配置.md)。旧配置不提供 `reranker` 时仍使用生成式筛选；JevHub 的 20 题真实结果已记录；官方服务尚未完成真实调用验证。兼容 `/rerank` 服务也仍可通过原 `reranker` 配置使用。

未提供 `tokenizer_file` 时，切分和 `query.context_budget` 明确以字符计数；提供本地 tokenizer JSON 后改用对应 token 计数，并将 tokenizer 文件摘要写入配置指纹。该预算针对证据正文，不代表服务的完整上下文上限。索引不下载模型、不偷偷截断原文。

英文使用 `examples/en.json`，填写英文生成、嵌入及 Jev 所需环境变量。每次重建必须使用新的 `collection_prefix`，防止破坏旧快照。其 `index/query/retrieve` 命令与中文相同，文档示例为 `examples/documents-en.jsonl`。默认档案锁定 Jev 1.13 原生路径，不把上游 1.14 冻结实验自动改称本项目的运行结果。

## HTTP 与 Python 入口

```bash
uv run vgrag-bi --config examples/zh.json --env-file .env serve
```

默认监听 `127.0.0.1:8088`。`POST /v1/query` 接收：

```json
{"namespace":"chinese_demo","language":"zh","question":"谁负责审批，期限多久？"}
```

`GET /health` 返回已配置的知识库和语言。HTTP 层仅供本地/受控服务集成，尚未实现多用户认证。Python 使用 `ChinesePipeline`、`EnglishUpstreamPipeline`、`SnapshotStore` 和 `Router`；多个知识库可通过 `api.create_app()` 注入。混合知识库必须明确选择 `zh/en`，`both` 当前显式返回不支持。

结果包含 `snapshot_id`、最终原文、已定位引文、缺失问点和逐阶段 trace。`complete` 表示本轮覆盖/校验流程通过，不代表经过标准答案评测；`partial` 保留缺口、回退或核验未通过的结果，`failed` 表示未能生成可用答复。

## 测试与评测恢复

```bash
uv run pytest -q
python evaluation/history/replay.py
```

测试使用合成材料、模拟模型与 Milvus Lite，阻止外部网络模型请求。中文解析回归包括 TXT、Markdown、DOCX 和 PDF；真实扫描 OCR 测试需要额外安装 Tesseract 及语言包，否则显式跳过。

`EvaluationRunner` 将生成和评分分开落盘：评分失败可以复用已有答案，`resume=True, retry_ids=[...]` 仅重跑指定题，配置变化禁止混合续跑；并发运行同一目录会被阻止，写断的末行可恢复。运行器接受显式注入的评分函数，没有隐式默认付费评测器。

历史千题的 79.1% 只属于 `evaluation/history/` 中的旧实验，不是本项目成绩。旧项目的 `.env` 未复制；本次 A/B 使用的数据、索引副本和原始逐题日志仅保存在 Git 忽略的 `.eval/` 目录。

## 文档

- [架构规划](docs/架构规划.md)
- [代码复用与重构清单](docs/代码复用与重构清单.md)
- [首版实现与验收](docs/首版实现与验收.md)
- [中文 20 题真实模型 A/B](docs/中文20题真实模型A-B.md)
- [中文链路修复与 Jev 接入](docs/中文链路修复与Jev接入.md)
- [中文 20 题修复后对照](docs/中文20题修复后对照.md)
- [中文切分基线](docs/中文切分基线.md)
- [历史评测记录](evaluation/history/README.md)
- [来源与许可](NOTICE.md)
