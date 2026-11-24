"""Resumable 1,000-question English evaluation with Jev disabled.

Runs 500 MuSiQue + 500 2WikiMultihopQA questions from the prepared manifest.
The runner is deliberately restartable: answers, triplets and scores are
append-only JSONL files and completed IDs are skipped on the next invocation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CHAT = "deepseek-ai/DeepSeek-V4.1-Flash"
LOCK = Lock()

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
def append(p, row):
    with LOCK:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
def digest(s): return hashlib.sha256(s.encode()).hexdigest()[:60]
def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def f1(pred, gold):
    a, b = norm(pred).split(), norm(gold).split()
    if not a or not b: return float(a == b)
    common = sum((min(a.count(x), b.count(x)) for x in set(a)), 0)
    if not common: return 0.0
    p, r = common / len(a), common / len(b)
    return 2*p*r/(p+r)
def exact(pred, answers): return any(norm(pred) == norm(x) for x in [answers] if isinstance(x, str))

def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    if not os.environ.get("VGRAG_CHAT_KEY"):
        raise RuntimeError("VGRAG_CHAT_KEY is required")
    os.environ["OPENAI_API_KEY"] = os.environ["VGRAG_CHAT_KEY"]

def make_docs(name, rows):
    docs, prepared = {}, []
    for q in rows:
        if name == "musique":
            mapping = {}
            for p in q["paragraphs"]:
                text = p["title"] + "\n\n" + p["paragraph_text"]
                pid = "p_" + digest(text)
                docs[pid] = {"id": pid, "source": pid, "text": text, "language": "en", "document_id": pid, "metadata": {"title": p["title"]}}
                mapping[str(p["idx"])] = pid
            gold = {mapping[str(p["idx"])] for p in q["paragraphs"] if p.get("is_supporting")}
            prepared.append({**q, "qid": str(q["id"]), "mapping": mapping, "gold_ids": sorted(gold), "gold_answers": [q.get("answer", ""), *q.get("answer_aliases", [])]})
        else:
            mapping = {}
            for title, sentences in q["context"]:
                text = title + "\n\n" + " ".join(sentences)
                pid = "p_" + digest(text); mapping[title] = pid
                docs[pid] = {"id": pid, "source": pid, "text": text, "language": "en", "document_id": pid, "metadata": {"title": title}}
            gold = {mapping[t] for t, _ in q.get("supporting_facts", []) if t in mapping}
            prepared.append({**q, "qid": str(q.get("_id", q.get("id"))), "mapping": mapping, "gold_ids": sorted(gold), "gold_answers": [q.get("answer", "")]})
    return prepared, list(docs.values())

def load_hipporag_triplets(data_dir, name):
    pattern = {
        "musique": "openie_musique_results_ner_gpt-3.5-turbo-1106_*.json",
        "2wikimultihopqa": "openie_2wikimultihopqa_results_ner_gpt-3.5-turbo-1106_*.json",
    }[name]
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HippoRAG triplet file matches {pattern}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    result = {}
    for doc in payload.get("docs", []):
        triples = [list(t[:3]) for t in doc.get("extracted_triples", [])
                   if isinstance(t, list) and len(t) == 3]
        result[(doc.get("title", ""), doc.get("text", ""))] = triples
    return result, files[-1]

def run_dataset(name, out, questions, docs, hippo_triples):
    from vector_graph_rag_bilingualism.config import EnglishConfig
    from vector_graph_rag_bilingualism.english import EnglishUpstreamPipeline
    from vector_graph_rag_bilingualism.models import Passage
    from vector_graph_rag_bilingualism.storage import SnapshotStore
    out.mkdir(parents=True, exist_ok=True)
    runtime = out / "standalone-index"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["VGRAG_HOME"] = str(out / "vgrag-home")
    (out / "vgrag-home").mkdir(exist_ok=True)
    settings = {"milvus_uri": os.getenv("VGRAG_MILVUS_URI", "http://127.0.0.1:19530"), "collection_prefix": "en_" + hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:16], "embedding_provider": "onnx", "embedding_model": "qdrant/bge-large-en-v1.5-onnx", "embedding_dimension": 1024, "llm_model": CHAT, "openai_base_url": "https://api.siliconflow.cn/v1", "final_top_k": 10, "use_llm_cache": False, "batch_size": 16, "milvus_consistency_level": "Bounded", "milvus_index_type": "AUTOINDEX"}
    store = SnapshotStore(runtime / "snapshots")
    pipe = EnglishUpstreamPipeline.connect(store, name, EnglishConfig(settings=settings), reranker=None, use_jev=False)
    if pipe.upstream.settings.uses_jev or getattr(pipe, "jev_accounting", None) is not None:
        raise RuntimeError("Jev unexpectedly enabled")
    fingerprint = pipe.fingerprint()
    saved = read_jsonl(out / "answers.jsonl")
    if any(row.get("pipeline_fingerprint") != fingerprint for row in saved):
        raise RuntimeError("Saved answers use a different or legacy retrieval policy; choose a new --out directory")
    up = pipe.upstream; up._triplet_extractor.cache = None; up._triplet_extractor.use_cache = False; up._answer_generator.cache = None; up._answer_generator.use_cache = False
    passages = [Passage(**x) for x in docs]
    triplet_rows = {p.id: {"id": p.id, "triplets": hippo_triples.get((p.metadata.get("title", ""), p.text.removeprefix(p.metadata.get("title", "") + "\n\n")), [])} for p in passages}
    matched = sum(bool(v["triplets"]) for v in triplet_rows.values())
    if matched == 0:
        raise RuntimeError(f"No HippoRAG triplets matched {name} passages")
    (out / "triplets.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in triplet_rows.values()), encoding="utf-8")
    print(name, "HippoRAG triplets", matched, "/", len(passages), flush=True)
    snap_file = out / "snapshot.json"
    if not snap_file.exists():
        snap = pipe.build(passages, extract_triplets=False, triplets={k: v["triplets"] for k, v in triplet_rows.items()})
        snap_file.write_text(json.dumps({"snapshot": snap}, ensure_ascii=False), encoding="utf-8")
    pipe.snapshot = json.loads(snap_file.read_text(encoding="utf-8"))["snapshot"]
    done = {x["qid"]: x for x in saved}
    for i, q in enumerate(questions, 1):
        if q["qid"] in done: continue
        start = time.time()
        result = pipe.query(q["question"])
        selected = {p.id for p in result.passages}; gold = set(q["gold_ids"])
        pred = result.answer or ""; best_f1 = max((f1(pred, a) for a in q["gold_answers"]), default=0.0)
        row = {"qid": q["qid"], "question": q["question"], "prediction": pred, "gold_answers": q["gold_answers"], "result": result.as_dict(), "selected_ids": sorted(selected), "gold_ids": sorted(gold), "recall_at_5": len({p.id for p in result.passages[:5]} & gold) / len(gold) if gold else 0.0, "support_recall": len(selected & gold) / len(gold) if gold else 0.0, "exact_match": any(norm(pred) == norm(a) for a in q["gold_answers"]), "f1": best_f1, "elapsed_s": round(time.time()-start, 2)}
        row["pipeline_fingerprint"] = fingerprint
        append(out / "answers.jsonl", row); done[q["qid"]] = row
        print(name, "answer", len(done), "/", len(questions), q["qid"], "EM", row["exact_match"], "R5", round(row["recall_at_5"], 3), flush=True)
    rows = list(done.values()); summary = {"dataset": name, "count": len(rows), "exact_match": sum(x["exact_match"] for x in rows)/len(rows), "f1": sum(x["f1"] for x in rows)/len(rows), "recall_at_5": sum(x["recall_at_5"] for x in rows)/len(rows), "support_recall": sum(x["support_recall"] for x in rows)/len(rows), "jev_enabled": False}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prepared", type=Path, default=ROOT / ".eval/english-nojev-1000"); ap.add_argument("--out", type=Path, default=ROOT / ".eval/english-nojev-1000-run"); ap.add_argument("--hipporag-data", type=Path, required=True); ap.add_argument("--dataset", choices=["musique", "2wikimultihopqa"]); args = ap.parse_args(); load_env()
    for name in ([args.dataset] if args.dataset else ("musique", "2wikimultihopqa")):
        raw = read_jsonl(args.prepared / f"{name}-questions.jsonl"); qs, docs = make_docs(name, raw)
        triples, source = load_hipporag_triplets(args.hipporag_data, name)
        print(name, "using", source.name, flush=True)
        run_dataset(name, args.out / name, qs, docs, triples)
    print("COMPLETE", flush=True)
if __name__ == "__main__": main()
