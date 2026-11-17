"""Resumable runs with durable generation checkpoints and targeted retries."""
from pathlib import Path
import json
import os
from .models import canonical, digest
from .metrics import diagnostics


def append(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("rb+") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n") or stream.read(1):
                    raise ValueError("Corrupt committed JSONL record") from None
                stream.seek(position)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                break
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return rows


class EvaluationRunner:
    def __init__(self, directory, pipeline, snapshot, *, judge=None, judge_identity=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pipeline, self.snapshot = pipeline, snapshot
        self.judge, self.judge_identity = judge, judge_identity
        if judge is not None and not judge_identity:
            raise ValueError("Judge identity is required for scored evaluation")

    def run(self, questions, *, resume=False, retry_ids=None):
        from .locks import exclusive_lock
        with exclusive_lock(self.directory / ".run.lock"):
            return self._run_locked(questions, resume=resume, retry_ids=retry_ids)

    def _run_locked(self, questions, *, resume=False, retry_ids=None):
        if len({q["id"] for q in questions}) != len(questions):
            raise ValueError("Duplicate question IDs")
        retry_ids = set(retry_ids or [])
        if not retry_ids.issubset({q["id"] for q in questions}):
            raise ValueError("Unknown targeted retry ID")
        if retry_ids and not resume:
            raise ValueError("Targeted retries require resume")
        manifest = {"schema": 1, "snapshot": self.snapshot, "pipeline": self.pipeline.fingerprint(),
                    "questions": digest(questions), "judge": self.judge_identity}
        path = self.directory / "manifest.json"
        if path.exists():
            if not resume or json.loads(path.read_text()) != manifest:
                raise ValueError("Existing evaluation manifest does not match")
        else:
            if resume:
                raise ValueError("Cannot resume without a manifest")
            temporary = path.with_suffix(".tmp")
            with temporary.open("w") as stream:
                stream.write(canonical(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        # Validate the requested snapshot before any model calls, even if all answers exist.
        self.pipeline.store.load(self.snapshot)
        pinned = self.pipeline.snapshot
        self.pipeline.snapshot = self.snapshot
        try:
            validator = getattr(self.pipeline, "_load", None) or self.pipeline._validate
            validator()
            return self._run(questions, retry_ids)
        finally:
            self.pipeline.snapshot = pinned

    def _run(self, questions, retry_ids):
        _, records = self.pipeline.store.load(self.snapshot)
        generations = self.directory / "generations.jsonl"
        scores = self.directory / "scores.jsonl"
        answers = {r["id"]: r for r in read_rows(generations)}
        grades = {(r["id"], r["attempt"]): r for r in read_rows(scores) if r.get("status") == "complete"}
        for q in questions:
            qid = q["id"]
            if retry_ids and qid not in retry_ids:
                continue
            answer = answers.get(qid)
            if answer is None or qid in retry_ids or answer.get("status") == "failed":
                attempt = (answer or {}).get("attempt", 0) + 1
                try:
                    result = self.pipeline.query(q["question"])
                    answer = {"id": qid, "attempt": attempt, "status": result.status, "result": result.as_dict()}
                    answer["diagnostics"] = diagnostics(q, answer["result"], records)
                except Exception as error:
                    answer = {"id": qid, "attempt": attempt, "status": "failed", "error": type(error).__name__}
                append(generations, answer)
                answers[qid] = answer
            key = (qid, answer["attempt"])
            if self.judge and answer["status"] != "failed" and key not in grades:
                try:
                    grade = self.judge(q, answer["result"])
                    row = {"id": qid, "attempt": answer["attempt"], "status": "complete", "grade": grade}
                    grades[key] = row
                except Exception as error:
                    row = {"id": qid, "attempt": answer["attempt"], "status": "failed", "error": type(error).__name__}
                append(scores, row)
        return {"questions": len(answers), "complete": sum(a["status"] == "complete" for a in answers.values()),
                "partial": sum(a["status"] == "partial" for a in answers.values()),
                "failed": sum(a["status"] == "failed" for a in answers.values()),
                "graded": sum((qid, a["attempt"]) in grades for qid, a in answers.items())}
