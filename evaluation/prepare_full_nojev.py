"""Prepare the fixed no-Jev English 1,000-question evaluation manifest.

This is intentionally preparation-only: it never calls a model or Jev.
The runner consumes the two JSONL files and may resume by question ID.
"""
import argparse, hashlib, json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-dataset", type=int, default=500)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = {}
    for name in ("musique", "2wikimultihopqa"):
        rows = sorted(load(args.upstream_data / f"{name}.json"), key=lambda r: str(r.get("id", r.get("_id"))))
        if len(rows) < args.per_dataset:
            raise ValueError(f"{name} has only {len(rows)} rows")
        selected[name] = rows[:args.per_dataset]
        write_jsonl(args.output / f"{name}-questions.jsonl", selected[name])
    manifest = {"schema": 1, "jev_enabled": False, "per_dataset": args.per_dataset,
                "datasets": {k: len(v) for k, v in selected.items()},
                "question_sha256": {k: hashlib.sha256((args.output / f"{k}-questions.jsonl").read_bytes()).hexdigest()
                                     for k in selected}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
