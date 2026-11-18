# 中英文统一使用 Milvus Standalone

项目固定部署 Milvus 2.6.9，客户端锁定为 PyMilvus 2.6.9。部署文件采用 [官方 embedded Standalone 布局](https://github.com/milvus-io/milvus/blob/v2.6.9/scripts/standalone_embed.sh)：一个 Milvus 容器，内嵌 etcd，使用本地持久化存储和 Docker 命名数据卷。本机原先的 MinIO 镜像已无法拉取，因此采用此官方支持的单容器布局。只向本机暴露 19530 和 9091 端口；此配置用于本地开发。

## 重启电脑后的启动步骤

Windows 需要虚拟机平台、WSL 2 和 Docker Desktop。Docker Desktop 使用 WSL 2 backend 和 Linux containers。启动 Docker Desktop 后，在项目根目录用 PowerShell 7 执行：

```powershell
docker info --format '{{.OSType}}'
docker compose up -d --wait
docker compose ps
Invoke-RestMethod http://127.0.0.1:9091/healthz
```

第一条应输出 `linux`。首次启动需要下载镜像。CPU 建议至少 4 核，为 WSL/Docker 留出至少约 8 GB 内存；16 GB 主机上先关闭其他占内存应用。数据库不需要 GPU。

停止使用 `docker compose stop`，恢复使用 `docker compose up -d --wait`。数据位于 `vgrag-bilingual` Compose 项目的命名卷内，停止容器不会删除数据。不要用删除数据卷的命令作为日常停止方式。

本机已启用虚拟化并完成部署：Docker Desktop 使用 WSL 2 / Linux containers，Ubuntu-22.04 和 Windows 均可访问上述健康检查地址。运行中的容器为 `vgrag-bilingual-standalone-1`，数据卷为 `vgrag-bilingual_milvus-data`。

## 应用连接

完整中英文依赖仍在 WSL/Linux 安装；固定上游的依赖声明包含 `milvus-lite`，这只是安装约束，正常运行不会再打开 Lite 文件。中文可以原生 Windows 运行。

```bash
uv sync --locked --extra english --extra api --extra documents --extra test
```

将下列项加入已有 `.env` 即可，不要覆盖现有模型密钥：

```dotenv
VGRAG_MILVUS_URI=http://127.0.0.1:19530
VGRAG_MILVUS_TOKEN=
```

连接优先级：`VGRAG_MILVUS_URI` → JSON profile 地址 → 本机 19530。中文配置位置为 `milvus.uri`，英文为 `upstream.milvus_uri`。没有配置不会退回本地搜索，`.db` 地址会被拒绝。认证信息仅通过环境变量读取。

通过 Docker Desktop 的 WSL Integration 在 WSL 中运行应用时，先确认 WSL 内也能访问 `http://127.0.0.1:9091/healthz`。若应用运行在其他容器/机器，需另外配置可达地址和相应网络；本 Compose 端口默认只绑定 Windows 本机。

中文 Collection 按快照内容命名为 `bi_<摘要>`；英文使用 `collection_prefix` 隔离，更新语料需要新前缀。两者可共用同一 Standalone。SQLite 的 `catalog.sqlite3` 仍保存快照、缓存和校验副本，不能删除。中文关系扩展和 BM25 保持在应用层执行，接入 Standalone 不改变这些算法。

## 新知识库

```bash
uv run vgrag-bi --config examples/zh.json --env-file .env index --documents examples/documents-zh.jsonl
uv run vgrag-bi --config examples/zh.json --env-file .env verify-index
uv run vgrag-bi --config examples/en.json --env-file .env index --documents examples/documents-en.jsonl
uv run vgrag-bi --config examples/en.json --env-file .env verify-index
```

`index` 会调用配置的抽取和 Embedding 模型。Milvus 不会替代模型服务；示例模型地址和密钥必须另行准备。仅迁移已有索引时使用下面的命令，不需要重新抽取或计算向量。

## 迁移已有中文快照

准备只含以下字段的本地 JSON 文件（填入真实 namespace 和快照目录）：

```json
{
  "language": "zh",
  "namespace": "原快照的 namespace",
  "data_dir": ".eval/你的评测目录/new-index",
  "milvus": {"uri": "http://127.0.0.1:19530"}
}
```

```bash
uv run vgrag-bi --config local-zh-migration.json --env-file .env --snapshot 原快照ID sync-milvus
```

未指定 `--snapshot` 时使用该 namespace 的活跃快照。命令将已有向量、原文、关系复制到 Milvus 并逐项校验，不调用模型，不修改旧快照 ID。重复执行只校验已有 Collection；发现内容不一致会报错，不覆盖它。后续查询应继续使用与旧快照模型和维度匹配的原应用配置。

Python 调用可使用 `pipeline.sync_milvus()`。中文评测脚本已在查询前调用这一操作，数据写入失败会终止，不会悄悄用本地搜索。

## 迁移已有英文 Lite 索引

在 WSL/Linux 中执行，先停止对源 Lite 文件的使用，并复制 Lite 文件作为迁移输入（Lite 打开文件时可能写入运行元数据）。准备英文目标 profile：namespace 与源快照一致，模型、维度和索引设置与原设置一致，只更改 `milvus_uri` 和 `collection_prefix`。使用新的 `data_dir` 和 Collection 前缀以保留原数据。

```bash
uv run vgrag-bi --config local-en-standalone.json --env-file .env --snapshot 原快照ID migrate-english --source-data-dir /原目录/snapshots --source-uri /原目录/index.db
```

命令复制三个 Collection 的向量、ID、关联字段，逐项核对内容摘要后创建并激活新快照，输出新快照 ID。源文件和源快照保留，目标已有同名且内容不同的 Collection 会被拒绝。迁移不调用 LLM/Embedding；它不能从只有段落快照、但丢失实体/关系的 Lite 文件中恢复完整英文图。后续查询使用目标 profile 和输出的新快照；不要把旧评测答案混入新的运行目录。

## 测试

离线算法、默认连接配置、迁移和防覆盖测试：

```bash
uv run pytest -q
```

数据库启动后，在安装了英文依赖的 WSL 中运行真实集成测试：

```bash
VGRAG_TEST_MILVUS_URI=http://127.0.0.1:19530 uv run pytest -q tests/test_milvus.py tests/test_english.py
```

这些集成测试使用模拟模型和随机测试 Collection，不产生付费模型调用，不修改现有知识库。未设置测试 URI 时会显式跳过，不能把跳过当作真实 Standalone 验证通过。

## 本机迁移记录

运行目录为 `.data/standalone/`（不提交到 Git）。`zh-migration.json` 和 `en-migration.json` 只用于搬运、校验数据，不是包含模型参数的问答配置。`migration-result.json` 记录迁移结果和新快照 ID。

中文来源为 `.eval/crud-flash-passage-50-20261117/new-index`，2,993 段；英文来源为 `.eval/musique20-20261113`，386 段及其实体、关系 Collection。原目录保留，迁移使用独立副本。后续问答仍需匹配原快照的模型设置和服务密钥。

最近英文 100 题评测的运行目录保留了答案和三元组，但未找到其原 `/tmp/vgrag-nojev` 索引，因此没有将该轮索引标记为已迁移，也没有自动调用模型重建。

完整性校验读取显式使用 Strong consistency，避免上游默认 Bounded consistency 在写入后立即读取时返回旧内容。

本机验证：中文真实数据库测试 1 项、英文真实数据库测试 9 项通过；Windows 离线回归 95 项通过、16 项按环境跳过（其中数据库集成测试已另外在 WSL 验证）。所有验证使用模拟模型或已有向量，没有发起付费模型请求。
