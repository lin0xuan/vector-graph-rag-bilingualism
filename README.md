# vector-graph-rag-bilingualism

准确率优先的中英文双链路 RAG。英文 `query()` / `retrieve()` 直接调用固定版本 Zilliz 上游，默认启用 Jev 1.13 的关系和段落两阶段重排；中文围绕问点覆盖、可定位证据、定向补查与回答核验编排，关系和段落评分也默认使用同一版 Jev。

中英文示例默认使用硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash` 和 Jev 1.13。提供 Python API、CLI、可选 HTTP 服务、不可变索引快照和评测恢复工具。Jev 默认连接官方 Typesafe，密钥为 `TYPESAFE_API_KEY`。JevHub 可通过显式配置启用。

2026-11-23 更新：英文检索不再使用本项目的生成式重排。未写 `reranker` 时也会把 `reranker_model` 设为 `jev-1.13.0`，并调用上游 `query()` / `retrieve()`。Jev 失败直接抛出。下方英文成绩属于更早的无 Jev 运行，不能算作当前默认链路的成绩。中文句子级选择仍用于标注重点证据，预算允许时也保留未被标注的高排名原文段落。

## 为什么单独建设中文链路

Milvus 能存储和检索中文向量，但 RAG 应用还需要在实体抽取、文本处理和证据检索中正确保留中文。一个具体例子是本项目固定依赖的上游 `vector-graph-rag` 提交 `1c034849fb36a104dadb1ddf967abcfe5bbd99b7`：其 [`processing_phrases()`](https://github.com/zilliztech/vector-graph-rag/blob/1c034849fb36a104dadb1ddf967abcfe5bbd99b7/src/vector_graph_rag/llm/extractor.py#L21-L31) 使用以下实体规范化逻辑：

```python
return re.sub(r"[^A-Za-z0-9 ]", " ", phrase.lower()).strip()
```

这会将 `北京大学`、`张三` 变成空字符串，将 `华为2024` 变成 `2024`。该函数实际用于 [`EntityExtractor.extract()` 的问题实体返回路径](https://github.com/zilliztech/vector-graph-rag/blob/1c034849fb36a104dadb1ddf967abcfe5bbd99b7/src/vector_graph_rag/llm/extractor.py#L316-L375)，包括缓存和实时模型结果。因此，即使模型正确抽出了中文实体，后处理也会丢失其中文名称，影响后续实体检索与图扩展；更换中文向量模型无法修复这一步的信息丢失。

本项目的 `language: "zh"` 使用独立的 [ChinesePipeline](src/vector_graph_rag_bilingualism/chinese.py)：问题实体校验为非空且出现在问题原文后原样保留；建库关系的主体、客体和引文也须来自原文，不经过上述清洗函数。可选 Jev 仅复用上游评分接口。这是建设并单独验证中文链路的一项具体依据；该示例针对上述固定版本，不代表上游所有检索路径都无法处理中文，也不能单独证明本项目的准确率更高，效果仍需以下评测验证。

## 最近完成的评测

以下模型回答与评分均使用 DeepSeek V4.1 Flash，均不使用 Jev。正确率表示模型评分器判定为完整正确的比例，不是人工审核结果。

| 链路与版本 | 数据集 / 题数 | 完整正确率 | Recall@5 | 最终证据召回率 |
|---|---|---:|---:|---:|
| 中文：保留完整段落 | CRUD / 50 | 64%（32/50） | 91.33% | 97.67% |
| 英文：两阶段改造前 | MuSiQue / 50 | 66%（33/50） | 80% | 80% |
| 英文：两阶段改造前 | 2WikiMultihopQA / 50 | 72%（36/50） | 79% | 79% |
| 英文：两阶段改造前合计 | 100 | 69%（69/100） | 79.5% | 79.5% |

- 中文为 25 道双文档题、25 道三文档题，使用冻结的 2,309 文档 / 2,993 段落语料。另有 18 题被评分器判为部分正确；按来源文档 ID 计算召回，命中该来源的任意片段即可计入。
- 英文复用 HippoRAG 预抽取的 OpenIE 三元组；每题搜索该数据集 50 题候选段落的共享并集，无逐题 passage-ID 限制。语料分别为 938 / 415 段，不是官方完整语料。样本按 ID 排序选取，MuSiQue 的这 50 题均为两跳题，不代表全部难度分布。
- 英文按标准支持段落 ID 计算召回。中英文指标粒度和题库不同，不能直接将百分比差距解释为语言能力差距；英文脚本的 EM/F1 也不是官方评分器口径。
- 上表英文成绩来自更早的无 Jev 运行。当前默认链路是上游 Jev，尚无按该链路复测的成绩。逐题原始记录位于本地 `.eval/`，不随 Git 发布；复核需要另行恢复该目录。

早期 20 题实验保留在[中文修复后对照](docs/中文20题修复后对照.md)和[Jev 启用前后对照](docs/中文20题Jev启用前后对照.md)，模型、样本和链路版本与上表不同。

2026-11-24 另有一批启用 JevHub 的中文 50 题，样本与上表无 Jev 的 50 题不重叠，不能并进上表。完整正确 36/50（72%），最终来源召回 97.33%，Recall@5 89.33%。过程与成绩见[新抽样评测记录](docs/中文50题新抽样评测记录说明.md)和[新抽样 Jev 评测](docs/中文50题新抽样Jev评测.md)。

## 安装

Python 3.11+，使用 `uv.lock` 固定依赖。英文依赖固定到上游 v0.4.1，提交 `1c034849fb36a104dadb1ddf967abcfe5bbd99b7`。这一版带上共享关系的实体链接、非法切分参数拒绝，以及 Milvus 强一致统计；不改变英文检索算法。下面的完整安装命令在 WSL2 / Linux 执行：

```bash
uv sync --locked --extra english --extra api --extra documents --extra test
uv run vgrag-bi --help
```

Windows 说明：中文核心、HTTP API、文档解析和离线测试可直接在 Windows 运行；评测锁也使用跨平台实现。英文和中文 Jev 额外依赖固定的上游 `vector-graph-rag`，其传递依赖 `milvus-lite` 当前没有 Windows wheel，因此这两项应在 WSL2/Linux 环境安装运行。若必须原生 Windows，需要先维护一个移除或替换该依赖的上游构建，不能只修改本项目的 extra 声明。

只开发中文核心时可省略 `english`。中英文现在默认共用 **Milvus Standalone**（`http://127.0.0.1:19530`）。中文将段落向量、原文及关系写入 Milvus，并由 Milvus 执行段落向量搜索；英文由固定上游维护实体、关系、段落三个 Collection。SQLite 仍保存不可变快照、校验信息和构建缓存，图扩展仍由应用代码执行。服务不可用时直接报错，不会自动回退到 SQLite 或 Lite。只有离线测试显式使用 `local_search=True`。

启动项目根目录的 `compose.yaml`：

```bash
docker compose up -d --wait
docker compose ps
```

Windows 先启用虚拟机平台、重启并启动 Docker Desktop 的 WSL 2 / Linux 容器引擎。完整部署、旧索引迁移及验证见 [Milvus Standalone 部署](docs/Milvus-Standalone.md)。

复跑英文 MuSiQue 评测时，运行 `uv sync --locked --extra english --extra onnx` 安装 ONNX 依赖。脚本优先使用 `.eval/musique20-20261113/vgrag-home/onnx-cache/` 内的模型，首次缺少模型时会联网下载。中文评测脚本使用 `evaluation` 额外依赖。复核历史评测时，先恢复双语版的 `.eval/` 和轻量版的 `.vgrag-eval/`。将两个仓库克隆到同一父目录即可运行 `uv run python evaluation/review_crud100.py`，也可以通过 `--light-root` 指定轻量版的位置。

## 配置与使用

```bash
cp .env.example .env
```

在 `.env` 填写自己的密钥，并复制 `examples/zh.json` 或 `examples/en.json` 为本地配置。聊天模型使用硅基流动地址，密钥为 `VGRAG_CHAT_KEY`；中文示例的向量服务仍需自行启动或替换。中英文共用 `VGRAG_MILVUS_URI` 和 `VGRAG_MILVUS_TOKEN`，其中 URI 环境变量优先于 JSON 配置；未设置时使用配置地址，再默认连接本机 19530。配置向量模型时保持名称、维度和已有索引一致。中文可在 `models` 中分别配置 `extract / plan / entities / relations / select / answer / verify`，未设置的角色继承 `default`。

以下索引和问答命令会访问你配置的模型服务，测试命令则不会：

```bash
uv run vgrag-bi --config examples/zh.json --env-file .env index --documents examples/documents-zh.jsonl
uv run vgrag-bi --config examples/zh.json --env-file .env query '谁负责示例项目的审批，办理期限多久？'
uv run vgrag-bi --config examples/zh.json --env-file .env verify-index
```

中文 `query.retrieval_mode` 支持 `single-stage` 和 `two-stage`，切换只影响查询策略，可以共用同一索引。`--snapshot ID` 固定已有快照。更新语料时重新输入该知识库完整文档清单，构建新快照后再原子切换；已完成的抽取/向量任务默认复用持久化检查点。源文件删除通过新清单排除该来源，旧快照仍可追溯。

中文示例已包含以下 Standalone 配置；英文使用 `upstream.milvus_uri`。正常入口拒绝 `.db` 文件地址：

```json
{"milvus": {"uri": "http://127.0.0.1:19530", "token_env": "VGRAG_MILVUS_TOKEN"}}
```

中文默认启用 Jev 1.13，给关系和段落打分；问点拆分、证据选择、回答和核验仍由聊天模型完成。不写 `reranker` 时，CLI 会使用官方 Typesafe 的 `jev-1.13.0`。缺少密钥会直接报错，不会悄悄改回生成式筛选。要改用 JevHub，复制 `examples/zh-jevhub.json`。旧的无 Jev 评测脚本直接构造管道，不经过这个默认值；CLI 需要关闭时写 `"use_jev": false`。兼容 `/rerank` 服务仍可通过显式 `reranker` 配置使用。可选 `on-demand` 模式见[双服务配置](docs/Jev双服务配置.md)，接入说明见[中文修复与 Jev 接入](docs/中文链路修复与Jev接入.md)。需要 `uv sync --extra jev`。

中文流程为问点拆分 → 原问题与各问点检索 → 关系筛选 → 句子证据标注 → 完整段落装入上下文 → 回答与核验，最多进行配置允许的两轮补查。段落按问点覆盖和检索排名在预算内装入；未找到可定位证据的问点会保留缺口标记，避免仅凭模型判断将其标为完成。

未提供 `tokenizer_file` 时，切分和 `query.context_budget` 明确以字符计数；提供本地 tokenizer JSON 后改用对应 token 计数，并将 tokenizer 文件摘要写入配置指纹。该预算针对证据正文，不代表服务的完整上下文上限。索引不下载模型、不偷偷截断原文。

英文示例使用与中文一致的 `deepseek-ai/DeepSeek-V4.1-Flash` 和硅基流动 `/v1` endpoint，聊天密钥复用 `VGRAG_CHAT_KEY`。Jev 默认使用官方 Typesafe，密钥为 `TYPESAFE_API_KEY`；`examples/en.json` 已写入 `reranker_model: jev-1.13.0` 和对应 `reranker`。不写 `reranker` 时，连接仍会默认打开同一 Jev 模型。要改用 JevHub，复制 `examples/en-jevhub.json`。缺少 Jev 密钥或请求失败会直接报错，不会改走另一套检索。只有评测脚本显式传入 `use_jev=False` 时，才使用上游自己的非 Jev 重排。每次重建必须使用新的 `collection_prefix`，防止破坏旧快照。其 `index/query/retrieve` 命令与中文相同，文档示例为 `examples/documents-en.jsonl`。英文配置里不要再写 `query` 调参块。

`query()` 和 `retrieve()` 都调用固定上游的同名方法。上游在 Jev 开启时抽取问题实体、检索并扩展关系、用 Jev 保留最多 64 条关系，再把关系段落与向量前 10 段交替合并、去重，最多 16 段后做段落评分，取 `final_top_k`。`query()` 用上游回答器生成答案，`retrieve()` 只返回段落。建库调用上游 `rebuild_documents()`。

显式 `filter` 原样传给上游，并同时约束引用映射。查询策略写入 pipeline 指纹；已有索引的配置指纹必须与当前上游设置一致，旧的无 Jev 或生成式重排结果不能混入续跑。当前没有在英文链路上增加迭代补查。

## HTTP 与 Python 入口

HTTP 服务需要 `api` 额外依赖（FastAPI / Uvicorn）。直接运行 Python 评测不依赖 FastAPI；未安装时 HTTP 接口测试会跳过。

```bash
uv run vgrag-bi --config examples/zh.json --env-file .env serve
```

默认监听 `127.0.0.1:8088`。`POST /v1/query` 接收：

```json
{"namespace":"chinese_demo","language":"zh","question":"谁负责审批，期限多久？"}
```

`GET /health` 返回已配置的知识库和语言。HTTP 层仅供本地/受控服务集成，尚未实现多用户认证。Python 使用 `ChinesePipeline`、`EnglishUpstreamPipeline`、`SnapshotStore` 和 `Router`；多个知识库可通过 `api.create_app()` 注入。混合知识库必须明确选择 `zh/en`，`both` 当前显式返回不支持。

结果包含 `snapshot_id`、最终原文和逐阶段 trace；中文还包含已定位引文与缺失问点。状态是执行与证据诊断，不代表答案正确：中文 `complete` 表示覆盖/核验流程通过，英文 `complete` 表示返回段落可无歧义映射到索引，英文重排回退另见 trace。`partial` / `failed` 的具体原因也记录在 trace 中。

## 测试与评测恢复

```bash
uv run pytest -q
python evaluation/history/replay.py
```

测试使用合成材料和模拟模型，阻止外部网络模型请求。算法单测显式使用离线本地检索；真实中英文存储集成测试连接 Standalone，需设置 `VGRAG_TEST_MILVUS_URI=http://127.0.0.1:19530`，未设置时跳过。英文集成测试还需要 `english` 额外依赖。测试仅创建和清理随机命名的测试 Collection，不接触已有知识库。中文解析回归包括 TXT、Markdown、DOCX 和 PDF；真实扫描 OCR 测试需要额外安装 Tesseract 及语言包，否则显式跳过。

主要评测入口：

| 脚本 | 用途 |
|---|---|
| `evaluation/prepare_full_nojev.py` | 从原始英文题库准备固定样本，`--per-dataset 50` 对应两套数据共 100 题 |
| `evaluation/run_full_nojev.py` | 使用 HippoRAG 三元组建立英文索引，调用当前问答链路并保存结果；`--hipporag-data` 指定三元组目录 |
| `evaluation/score_english_saved.py` | 对已保存的英文回答做模型语义评分，不重新生成答案 |
| `evaluation/run_crud_50_nojev.py` | 使用已准备的 CRUD 50 题及冻结索引运行中文评测 |

模型调用脚本会产生接口费用。各脚本通过 `--help` 查看所需路径；中文脚本依赖历史样本、索引和本地模型缓存，不是下载仓库后即可直接运行的独立题库。中文评测在查询前将冻结快照同步到 Standalone，复用已有向量和关系。英文 `run_full_nojev.py` 将快照保存到输出目录的 `standalone-index/snapshots`，向量数据保存在 Standalone 的 Docker 数据卷中，Collection 前缀按输出目录隔离。旧 `/tmp/vgrag-nojev/` 和 Lite 文件不会被自动删除或迁移。`run_musique20.py` 默认使用新的 `.eval/musique20-standalone` 输出，可通过 `VGRAG_MUSIQUE_OUT` 指定。切换后应使用独立评测输出，不把旧本地检索成绩当作 Standalone 成绩；备份需同时保留服务器数据卷、快照和输出。

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
