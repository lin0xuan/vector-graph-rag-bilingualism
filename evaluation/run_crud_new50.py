"""Run 50 previously unused CRUD questions on the frozen Chinese index.

The sample is 25 two-document and 25 three-document questions drawn with seed
20261123 from the prepared 1,000-question pool, excluding every question id
already present in earlier .eval question files. Dense retrieval uses the
frozen vectors with exact local cosine because this entry does not require
Milvus. Relation and passage order use JevHub jev-1.13.0.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
import shutil
import threading
import time

import compare_light as base
from retest_jev import JUDGE_PROMPT
from run_crud_50_nojev import load_chat_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="Directory containing catalog.sqlite3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("Use 1 or 2 workers")
    if args.output.resolve() == args.index.resolve():
        parser.error("Write results to a separate directory")
    args.output.mkdir(parents=True, exist_ok=True)
    from vector_graph_rag_bilingualism.locks import exclusive_lock
    with exclusive_lock(args.output / "run.lock"):
        execute(args)


def execute(args):
    from fastembed import TextEmbedding
    from tokenizers import Tokenizer
    from vector_graph_rag_bilingualism import ChineseConfig, ChinesePipeline, SnapshotStore
    from vector_graph_rag_bilingualism.cli import load_env
    from vector_graph_rag_bilingualism.jev import JevReranker
    from vector_graph_rag_bilingualism.metrics import diagnostics
    from vector_graph_rag_bilingualism.providers import PROMPTS

    load_env(args.env_file)
    sample = base.rows(args.questions)
    if len(sample) != 50 or Counter(len(q["sources"]) for q in sample) != {2: 25, 3: 25}:
        raise ValueError("Expected 25 two-document and 25 three-document questions")
    index = args.output / "new-index"
    index.mkdir(exist_ok=True)
    if not (index / "catalog.sqlite3").exists():
        shutil.copy2(args.index / "catalog.sqlite3", index / "catalog.sqlite3")
    import sqlite3
    with sqlite3.connect(index / "catalog.sqlite3") as db:
        namespace, snapshot = db.execute("SELECT namespace, snapshot FROM active").fetchone()
    store = SnapshotStore(index)
    stored, records = store.load(snapshot)
    if stored["namespace"] != namespace:
        raise ValueError("Active snapshot namespace mismatch")
    sources = {row["passage"]["source"] for row in records}
    if any(not set(q["sources"]) <= sources for q in sample):
        raise ValueError("A sampled question uses a document outside the frozen index")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    embedding = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir=str(args.embedding_cache), threads=2)
    tokenizer = Tokenizer.from_str(embedding.model.tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    embedder = base.SharedEmbedding(type("Light", (), {"model": embedding, "embed_lock": threading.Lock()})())
    if stored["config"]["embedding"] != embedder.identity():
        raise ValueError("Frozen index embedding does not match bge-small-zh-v1.5")
    client = base.RealRequests(load_chat_key(args.env_file), args.output, tokenizer)
    model = base.RealJsonModel(client, PROMPTS)

    def observe(event):
        base.append(args.output / "jev-http.jsonl", {"phase": "evaluation", **event})

    jev = JevReranker(service="jevhub", request_observer=observe)
    config = ChineseConfig()
    manifest = {
        "schema": 1, "experiment": "crud-new-50-jevhub", "questions": 50,
        "sample_sha256": base.sha(args.questions),
        "sample_selection": "Seed 20261123; 25 two-document and 25 three-document questions absent from earlier .eval question files",
        "new_snapshot": snapshot, "index_content_hash": stored["content_hash"],
        "documents": len(sources), "passages": len(records),
        "query_model": base.CHAT, "judge_model": base.JUDGE, "temperature": 0,
        "enable_thinking": False, "answer_max_tokens": 2000,
        "jev_enabled": True, "jev": jev.identity(), "retrieval": "local-exact-frozen-vectors",
        "new_config": asdict(config), "model": model.identity(),
    }
    target = args.output / "manifest.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Configuration changed; use a separate output directory")
    base.atomic(target, manifest)
    question_copy = args.output / "questions.jsonl"
    if question_copy.exists() and base.sha(question_copy) != manifest["sample_sha256"]:
        raise ValueError("Question file changed")
    if not question_copy.exists():
        shutil.copy2(args.questions, question_copy)
    pipeline = ChinesePipeline(
        store, namespace, embedder, model, config=config, snapshot=snapshot, local_search=True,
        reranker=jev, relation_reranker=jev,
        count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
        tokenizer_id="bge-small-zh-v1.5-frozen-tokenizer")
    latest = {row["id"]: row for row in base.rows(args.output / "generations.jsonl")}
    grades = {row["id"]: row for row in base.rows(args.output / "grades.jsonl") if row.get("grade")}

    def work(question):
        base.LOCAL.qid = question["id"]
        base.LOCAL.arm = "jev"
        row = latest.get(question["id"])
        if row is None:
            started = time.monotonic()
            try:
                result = pipeline.query(question["question"])
                value = result.as_dict()
                stats = diagnostics(question, value, records)
                gold = set(question["sources"])
                candidate = stats["candidate_sources"] or {"recall": 0, "all": False, "hit": []}
                final = stats["final_sources"]
                row = {
                    "id": question["id"], "arm": "jev", "question": question["question"],
                    "answer": result.answer, "status": result.status,
                    "citations": [{"id": p.id, "source": p.source, "text": p.text} for p in result.passages],
                    "candidate_recall": candidate["recall"], "candidate_all_sources": candidate["all"],
                    "candidate_matched_sources": candidate["hit"], "final_recall": final["recall"],
                    "all_sources": final["all"], "final_matched_sources": final["hit"],
                    "recall_at_5": len(gold & {p.source for p in result.passages[:5]}) / len(gold),
                    "trace": value["trace"], "evidence": value["evidence"],
                    "missing_slots": result.missing_slots, "snapshot_id": snapshot,
                }
            except Exception as error:
                row = {
                    "id": question["id"], "arm": "jev", "question": question["question"],
                    "answer": None, "status": "query_error", "error": type(error).__name__,
                    "citations": [], "candidate_recall": 0, "candidate_all_sources": False,
                    "final_recall": 0, "all_sources": False, "recall_at_5": 0,
                }
            row.update(documents=len(question["sources"]), query_ms=round((time.monotonic() - started) * 1000))
            base.append(args.output / "generations.jsonl", row)
            print(f"Generated {question['id']}: {row['status']}, final sources={row['final_recall']:.2f}", flush=True)
        if question["id"] not in grades:
            if not (row.get("answer") or "").strip():
                graded = {"id": question["id"], "arm": "jev",
                          "grade": {"accuracy": "wrong", "grounded": False, "reason": "未生成可评答案。"},
                          "rule_applied": "empty_answer"}
            else:
                payload = {"question": question["question"], "reference": question["reference"],
                           "answer": row["answer"], "evidence": row["citations"]}
                try:
                    answer, _ = client.ask("judge", JUDGE_PROMPT, payload, model=base.JUDGE,
                                           json_mode=True, max_tokens=1000)
                    grade = json.loads(answer)
                    if grade.get("accuracy") not in ("correct", "partial", "wrong") or type(grade.get("grounded")) is not bool:
                        raise ValueError("Invalid grade")
                    graded = {"id": question["id"], "arm": "jev", "grade": grade}
                except Exception as error:
                    graded = {"id": question["id"], "arm": "jev", "grade": None, "error": type(error).__name__}
            base.append(args.output / "grades.jsonl", graded)
            print(f"Graded {question['id']}: {(graded.get('grade') or {}).get('accuracy', 'failed')}", flush=True)
        return question["id"]

    print(f"Frozen snapshot {snapshot}; 50 new CRUD questions; JevHub; local exact retrieval; workers={args.workers}.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task in as_completed(pool.submit(work, question) for question in sample):
            print("Completed: " + task.result(), flush=True)
    for _ in range(2):
        latest = {row["id"]: row for row in base.rows(args.output / "generations.jsonl")}
        grades = {row["id"]: row for row in base.rows(args.output / "grades.jsonl") if row.get("grade")}
        pending = [question for question in sample if question["id"] in latest and question["id"] not in grades]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for task in as_completed(pool.submit(work, question) for question in pending):
                print("Completed: " + task.result(), flush=True)
    graded = [row for row in base.rows(args.output / "grades.jsonl") if row.get("grade")]
    generated = base.rows(args.output / "generations.jsonl")
    counts = Counter(row["grade"]["accuracy"] for row in graded)
    recalls = [row["final_recall"] for row in generated if row.get("final_recall") is not None]
    summary = {
        "graded": len({row["id"] for row in graded}),
        "accuracy": dict(counts),
        "correct_rate": counts["correct"] / len(sample),
        "mean_final_recall": sum(recalls) / len(recalls) if recalls else 0,
        "mean_recall_at_5": sum(row.get("recall_at_5") or 0 for row in generated) / len(sample),
    }
    base.atomic(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    client.http.close()


if __name__ == "__main__":
    main()
