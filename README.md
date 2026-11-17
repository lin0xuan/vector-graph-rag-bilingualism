# vector-graph-rag-bilingualism

准确率优先的中英文双链路 RAG。英文复用固定版本 Zilliz 的索引、向量检索和图扩展，增加生成式关系与原文段落两阶段重排；中文围绕问点覆盖、可定位证据、定向补查与回答核验编排。

中英文示例默认使用硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`，均默认不启用 Jev。提供 Python API、CLI、可选 HTTP 服务、不可变索引快照和评测恢复工具。Jev 官方服务和 JevHub 可通过显式配置启用。

2026-11-17 更新：英文 `query()` / `retrieve()` 共用两阶段检索流程，合并图检索和向量检索段落后按原文排序；中文句子级选择用于标注重点证据，预算允许时也保留未被标注的高排名原文段落。英文新策略已经通过离线回归，但尚未完成真实题目复测；下方英文成绩属于修改前版本。

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
- 本次新增英文段落重排、宽关系候选和返回校验尚无新成绩，不把旧结果归到新实现。逐题原始记录位于本地 `.eval/`，不随 Git 发布；复核需要另行恢复该目录。

早期 20 题实验保留在[中文修复后对照](docs/中文20题修复后对照.md)和[Jev 启用前后对照](docs/中文20题Jev启用前后对照.md)，模型、样本和链路版本与上表不同。

## 安装

Python 3.11+，使用 `uv.lock` 固定依赖。英文依赖固定到上游提交 `47590ad4eab5838cdcfec94f8a53ebfe70ebe330`。下面的完整安装命令在 WSL2 / Linux 执行：

```bash
uv sync --locked --extra english --extra api --extra documents --extra test
uv run vgrag-bi --help
```

Windows 说明：中文核心、HTTP API、文档解析和离线测试可直接在 Windows 运行；评测锁也使用跨平台实现。英文和中文 Jev 额外依赖固定的上游 `vector-graph-rag`，其传递依赖 `milvus-lite` 当前没有 Windows wheel，因此这两项应在 WSL2/Linux 环境安装运行。若必须原生 Windows，需要先维护一个移除或替换该依赖的上游构建，不能只修改本项目的 extra 声明。

只开发中文核心时可省略 `english`。中文默认使用 SQLite 中的精确向量检索进行开发；配置 `milvus` 后使用 Milvus 查询，并校验镜像实际内容。英文存储由上游 Milvus 负责。

复跑英文 MuSiQue 评测时，运行 `uv sync --locked --extra english --extra onnx` 安装 ONNX 依赖。脚本优先使用 `.eval/musique20-20261113/vgrag-home/onnx-cache/` 内的模型，首次缺少模型时会联网下载。中文评测脚本使用 `evaluation` 额外依赖。复核历史评测时，先恢复双语版的 `.eval/` 和轻量版的 `.vgrag-eval/`。将两个仓库克隆到同一父目录即可运行 `uv run python evaluation/review_crud100.py`，也可以通过 `--light-root` 指定轻量版的位置。

## 配置与使用

```bash
cp .env.example .env
```

在 `.env` 填写自己的密钥，并复制 `examples/zh.json` 或 `examples/en.json` 为本地配置。聊天模型使用硅基流动地址，密钥为 `VGRAG_CHAT_KEY`；中文示例的向量服务和英文示例的 Milvus 地址指向本机，需自行启动或替换。配置向量模型时保持名称、维度和已有索引一致。中文可在 `models` 中分别配置 `extract / plan / entities / relations / select / answer / verify`，未设置的角色继承 `default`。

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

中文默认不启用 Jev，使用生成式关系筛选、证据选择和回答核验；只有配置顶层 `reranker` 才会创建远程重排客户端。中文 Jev 支持固定上游 `/systemone` 协议；见 `examples/zh-jev.json` 和[中文修复与 Jev 接入](docs/中文链路修复与Jev接入.md)。需要 `uv sync --extra jev`。显式启用 Jev 而不设置 `mode` 时，不加“每题一次”的限制；中文另有可选 `on-demand` 模式，其候选阈值和调用次数可配置，详见[双服务配置](docs/Jev双服务配置.md)。兼容 `/rerank` 服务也可通过显式 `reranker` 配置使用。

中文流程为问点拆分 → 原问题与各问点检索 → 关系筛选 → 句子证据标注 → 完整段落装入上下文 → 回答与核验，最多进行配置允许的两轮补查。段落按问点覆盖和检索排名在预算内装入；未找到可定位证据的问点会保留缺口标记，避免仅凭模型判断将其标为完成。

未提供 `tokenizer_file` 时，切分和 `query.context_budget` 明确以字符计数；提供本地 tokenizer JSON 后改用对应 token 计数，并将 tokenizer 文件摘要写入配置指纹。该预算针对证据正文，不代表服务的完整上下文上限。索引不下载模型、不偷偷截断原文。

英文示例使用与中文一致的 `deepseek-ai/DeepSeek-V4.1-Flash` 和硅基流动 `/v1` endpoint，密钥复用 `VGRAG_CHAT_KEY`。默认英文链路也不启用 Jev；只有显式传入 `reranker` 配置才启用，连接或请求失败会回退到上游默认路径。每次重建必须使用新的 `collection_prefix`，防止破坏旧快照。其 `index/query/retrieve` 命令与中文相同，文档示例为 `examples/documents-en.jsonl`。

英文默认无 Jev 链路使用两阶段生成式重排：图扩展后由当前配置的模型排序，保留最多 64 条关系；轮流取关系对应段落，与普通向量检索前 10 段交替合并、按文本去重，最多保留 16 段；再让同一模型读取来源标题和完整原文段落，统一排序，取 `final_top_k` 段。`query()` 和 `retrieve()` 使用相同的检索逻辑，只有前者生成答案。这里借鉴 Jev 的候选合并方式，评分器使用普通大模型，不代表复现 Jev 的效果。启用 Jev 时仍走固定上游的 Jev 实现。

可通过英文顶层 `query` 配置 `relation_limit`（64）、`dense_top_k`（10）、`passage_limit`（16）、`batch_chars`（60000 字符）和 `max_output_tokens`（4096）。向量数量和合并上限至少为 `final_top_k`，但语料不足时不保证填满。模型调用不设置空行停止条件；检查结束原因、JSON、编号范围、去重和返回数量，失败重试一次后保留该批输入顺序。空编号列表在有候选时视为格式不符，因为此处要求排序而不是删除候选。大输入按完整候选分批，批次结果交替合并；超预算的单个候选保留原文并跳过该批模型调用，不静默截断。字符预算不是模型 token 上限。

显式 `filter` 同时约束图检索、向量检索和提供给模型的来源信息。Trace 记录两阶段原始模型回复、结束原因、重试/回退和候选顺序，原始回复可能含文档信息，按评测数据保存。查询策略写入 pipeline 指纹；已有索引可继续使用，旧评测结果不能混入新策略续跑。当前没有增加迭代补查或别名合并。

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

测试使用合成材料、模拟模型与 Milvus Lite，阻止外部网络模型请求。中文解析回归包括 TXT、Markdown、DOCX 和 PDF；真实扫描 OCR 测试需要额外安装 Tesseract 及语言包，否则显式跳过。

主要评测入口：

| 脚本 | 用途 |
|---|---|
| `evaluation/prepare_full_nojev.py` | 从原始英文题库准备固定样本，`--per-dataset 50` 对应两套数据共 100 题 |
| `evaluation/run_full_nojev.py` | 使用 HippoRAG 三元组建立英文索引，调用当前问答链路并保存结果；`--hipporag-data` 指定三元组目录 |
| `evaluation/score_english_saved.py` | 对已保存的英文回答做模型语义评分，不重新生成答案 |
| `evaluation/run_crud_50_nojev.py` | 使用已准备的 CRUD 50 题及冻结索引运行中文评测 |

模型调用脚本会产生接口费用。各脚本通过 `--help` 查看所需路径；中文脚本依赖历史样本、索引和本地模型缓存，不是下载仓库后即可直接运行的独立题库。英文脚本的索引运行目录在 WSL `/tmp/vgrag-nojev/`，备份时需同时保留索引、快照和输出，不能只复制答案文件。新查询策略需要独立评测输出，旧答案缺少匹配的 pipeline 指纹时拒绝混合续跑。

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
