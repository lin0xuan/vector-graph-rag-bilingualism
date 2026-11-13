from collections import Counter
from hashlib import sha256
from vector_graph_rag_bilingualism.models import Passage


class FakeEmbedding:
    dimension = 16
    def identity(self):
        return {"model": "offline-test-hash", "dimension": self.dimension}
    def embed(self, texts):
        result = []
        for text in texts:
            values = [0.01] * self.dimension
            for ch in text:
                values[sha256(ch.encode()).digest()[0] % self.dimension] += 1
            norm = sum(v*v for v in values) ** 0.5
            result.append([v/norm for v in values])
        return result
    def embed_batch(self, texts, **kwargs):
        return self.embed(texts)


class FakeModel:
    def __init__(self):
        self.calls = Counter()
        self.fail_entities = False
        self.fail_extract = False
        self.invalid_quote = False
        self.omit_first_answer = False
        self.fail_select = False
    def identity(self):
        return {"model": "offline-fixture-v1"}
    def call(self, task, data, trace):
        self.calls[task] += 1
        if task == "extract":
            if self.fail_extract:
                raise TimeoutError()
            return {"relations": []}
        if task == "entities":
            if self.fail_entities:
                raise TimeoutError()
            return {"entities": []}
        if task == "plan":
            return {"slots": [{"id": "s1", "question": "谁负责审批"}, {"id": "s2", "question": "期限多久"}]}
        if task == "select":
            if self.fail_select:
                raise TimeoutError()
            items = []
            for p in data["candidates"]:
                for sid, marker in (("s1", "审批"), ("s2", "期限")):
                    if sid in {s["id"] for s in data["slots"]} and any(marker in sentence["text"] for sentence in p["sentences"]):
                        items.append({"slot_id": sid, "passage_id": p["id"], "sentence_ids": [999] if self.invalid_quote else [sentence["id"] for sentence in p["sentences"] if marker in sentence["text"]]})
            return {"evidence": items}
        if task == "answer":
            evidence = data["evidence"] or [{"quote": p["text"], "passage_id": p["id"]} for p in data["passages"]]
            if self.omit_first_answer and "previous_answer" not in data:
                evidence = evidence[:1]
            return {"answer": "\n".join(e["quote"] + "[" + e["passage_id"] + "]" for e in evidence)}
        if task == "verify":
            ok = "期限" in data["answer"]
            return {"ok": ok, "issues": [] if ok else ["遗漏期限"]}
        if task == "relations":
            return {"ids": [r["id"] for r in data["relations"]]}
        raise AssertionError(task)


def corpus():
    return [Passage("a", "policy", "张某负责审批。", "zh", "policy"),
            Passage("b", "time", "办理期限为三个工作日。", "zh", "time"),
            Passage("c", "noise", "这是一份无关的气象资料。", "zh", "noise")]
