"""Small operational CLI; no credentials or network activity on --help."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from .chinese import ChinesePipeline
from .config import ChineseConfig, EnglishConfig, EnglishQueryConfig, ModelService
from .english import EnglishUpstreamPipeline
from .ingestion import ingest_file, ingest_text
from .models import Passage, canonical, digest
from .providers import Embeddings, JsonModel, Reranker, SelectiveReranker
from .storage import SnapshotStore


def load_env(path):
    # Deliberately no shell evaluation or interpolation.
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isidentifier():
            raise ValueError("Invalid env-file line")
        os.environ.setdefault(key, value.strip().strip('"\''))


def connect(config, snapshot=None):
    store = SnapshotStore(config.get("data_dir", ".data"))
    namespace = config["namespace"]
    language = config["language"]
    if language == "en":
        english = EnglishConfig(config["upstream"], query=EnglishQueryConfig(**config.get("query", {})))
        pipeline = EnglishUpstreamPipeline.connect(store, namespace, english, snapshot=snapshot,
                                                   reranker=config.get("reranker"))
    elif language == "zh":
        embedding = config["embedding"]
        embedder = Embeddings(ModelService(**embedding["service"]), embedding["dimension"])
        model = JsonModel({k: ModelService(**v) for k, v in config["models"].items()})
        mirror = None
        local_search = config.get("local_search", False)
        if type(local_search) is not bool:
            raise ValueError("local_search must be a boolean")
        if local_search and "milvus" in config:
            raise ValueError("local_search cannot be combined with milvus configuration")
        if not local_search:
            from .milvus import MilvusMirror
            mirror = MilvusMirror.connect(config.get("milvus"))
        count = len
        tokenizer_id = "characters-v1"
        if config.get("tokenizer_file"):
            from tokenizers import Tokenizer
            from hashlib import sha256
            tokenizer_id = "sha256:" + sha256(Path(config["tokenizer_file"]).read_bytes()).hexdigest()
            tokenizer = Tokenizer.from_file(config["tokenizer_file"])
            tokenizer.no_truncation()
            tokenizer.no_padding()
            count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
        reranker = None
        relation_reranker = None
        if config.get("reranker"):
            options = dict(config["reranker"])
            provider = options.pop("provider", "compatible-http")
            mode = options.pop("mode", "always")
            min_candidates = options.pop("min_candidates", 48)
            max_calls_per_query = options.pop("max_calls_per_query", 1)
            if provider == "jev":
                from .jev import JevReranker
                try:
                    reranker = relation_reranker = JevReranker(**options)
                except Exception:
                    # Jev is optional. Missing credentials or an unavailable
                    # service must leave the default local/generative path usable.
                    reranker = relation_reranker = None
            elif provider == "compatible-http":
                reranker = Reranker(ModelService(**options))
            else:
                raise ValueError("Unknown reranker provider")
            if mode == "on-demand":
                selective = SelectiveReranker(reranker, min_candidates=min_candidates,
                                              max_calls_per_query=max_calls_per_query)
                reranker = relation_reranker = selective
            elif mode != "always":
                raise ValueError("Unknown reranker mode")
        pipeline = ChinesePipeline(store, namespace, embedder, model, config=ChineseConfig(**config.get("query", {})),
                                   snapshot=snapshot, mirror=mirror, count_tokens=count,
                                   reranker=reranker, relation_reranker=relation_reranker,
                                   tokenizer_id=tokenizer_id, local_search=local_search)
    else:
        raise ValueError("language must be zh or en")
    return pipeline


def load_documents(path, config, pipeline):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    passages = []
    for row in rows:
        if not row.get("source"):
            raise ValueError("Every document needs a stable source ID")
        if config["language"] == "zh":
            options = {**config.get("splitter", {}), "count": pipeline.count_tokens,
                       "document_id": row.get("document_id"), "title": row.get("title")}
            if "path" in row:
                from .ingestion.documents import parse_file
                from .ingestion import passages_from_blocks
                passages.extend(passages_from_blocks(row["source"], parse_file(row["path"]), **options))
            else:
                passages.extend(ingest_text(row["source"], row["text"], **options))
        else:
            from vector_graph_rag.models import Document
            from vector_graph_rag.loaders.chunker import TextChunker
            # English input is explicit text; richer file loading remains upstream's concern.
            if "text" not in row:
                raise ValueError("English CLI documents require text; use upstream loaders before indexing files")
            docs = TextChunker(**config.get("splitter", {})).chunk(Document(page_content=row["text"]))
            for index, doc in enumerate(docs):
                pid = digest([row["source"], index, doc.page_content])
                passages.append(Passage(pid, row["source"], doc.page_content, "en",
                                        row.get("document_id") or row["source"], doc.metadata))
    return passages


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent English and Chinese RAG pipelines")
    parser.add_argument("--config", required=True, help="JSON profile (credentials referenced by env name)")
    parser.add_argument("--env-file", help="Optional local environment file; never executed as shell")
    parser.add_argument("--snapshot", help="Pin an existing snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--documents", required=True, help="JSONL source/text or Chinese source/path")
    for command in ("query", "retrieve"):
        item = sub.add_parser(command)
        item.add_argument("question")
        item.add_argument("--filter", help="Native English Milvus metadata filter")
    sub.add_parser("verify-index")
    sub.add_parser("sync-milvus", help="Copy an existing Chinese snapshot to Standalone without model calls")
    migrate = sub.add_parser("migrate-english", help="Copy an existing English Milvus index without model calls")
    migrate.add_argument("--source-data-dir", required=True, help="Directory containing the source catalog.sqlite3")
    migrate.add_argument("--source-uri", required=True, help="Existing source Milvus Lite file (run in WSL/Linux)")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8088)
    args = parser.parse_args(argv)
    pipeline = None
    try:
        if args.env_file:
            load_env(args.env_file)
        config = json.loads(Path(args.config).read_text())
        if args.command == "migrate-english":
            from .migration import migrate_english
            result = migrate_english(config, args.source_data_dir, args.source_uri, args.snapshot)
            print(canonical(result))
            return
        if args.command == "sync-milvus":
            from .migration import sync_chinese
            result = sync_chinese(config, args.snapshot)
            print(canonical(result))
            return
        pipeline = connect(config, args.snapshot)
        if args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app({config["namespace"]: {config["language"]: pipeline}}), host=args.host, port=args.port)
            return
        if args.command == "index":
            if args.snapshot:
                raise ValueError("Do not supply --snapshot when building a new index")
            passages = load_documents(args.documents, config, pipeline)
            kwargs = {"ingestion_config": config.get("splitter", {})} if config["language"] == "zh" else {}
            result = {"snapshot": pipeline.build(passages, **kwargs), "passages": len(passages)}
        elif args.command == "verify-index":
            snapshot, records = pipeline._load() if config["language"] == "zh" else pipeline._validate()
            result = {"snapshot": snapshot, "records": len(records), "status": "verified"}
        else:
            if args.filter and config["language"] != "en":
                raise ValueError("Chinese namespace isolation is supported; raw Milvus filters are English-only")
            kwargs = {"filter": args.filter} if config["language"] == "en" else {}
            result = getattr(pipeline, args.command)(args.question, **kwargs).as_dict()
        print(canonical(result))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        # Provider errors are redacted by provider adapters. Avoid arbitrary upstream exception bodies.
        parser.exit(2, f"{type(error).__name__}: operation failed; check Standalone availability, profile and index integrity. "
                    "For an old Chinese snapshot, run sync-milvus first.\n")
    finally:
        if pipeline:
            client = pipeline.client if isinstance(pipeline, EnglishUpstreamPipeline) else getattr(pipeline.mirror, "client", None)
            if client:
                client.close()


if __name__ == "__main__":
    main()
