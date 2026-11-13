"""Dense, BM25 and bounded graph retrieval; ranks retain their channel provenance."""
from collections import Counter, defaultdict
import math
import re
from .models import vector, digest


def terms(text):
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[\u3400-\u9fff]+", text.lower())
    result = []
    for word in words:
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            result.extend(word)
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            result.append(word)
    return result


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)))


class Retriever:
    def __init__(self, records, embedder, *, mirror=None, snapshot="", relation_ranker=None):
        self.records = {r["passage"]["id"]: r for r in records}
        self.embedder, self.mirror, self.snapshot = embedder, mirror, snapshot
        self.relation_ranker = relation_ranker
        self.counts = {pid: Counter(terms(r["passage"]["text"])) for pid, r in self.records.items()}
        self.df = Counter(term for counts in self.counts.values() for term in counts)
        self.query_vectors = {}
        self.average = sum(sum(c.values()) for c in self.counts.values()) / max(len(records), 1)

    def lexical(self, question, limit):
        query = set(terms(question))
        scores = {}
        n = len(self.counts)
        for pid, counts in self.counts.items():
            length = sum(counts.values())
            score = 0.0
            for term in query:
                frequency = counts[term]
                if frequency:
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / max(self.average, 1)))
            if score:
                scores[pid] = score
        return sorted(scores, key=lambda pid: (-scores[pid], pid))[:limit]

    def dense(self, question, limit):
        dimension = len(next(iter(self.records.values()))["vector"])
        values = self.embedder.embed([question])
        if len(values) != 1:
            raise ValueError("Query embedding response count mismatch")
        query = vector(values[0], dimension)
        self.query_vectors[question] = query
        if self.mirror:
            return self.mirror.search(self.snapshot, query, limit)
        return sorted(self.records, key=lambda pid: (-cosine(query, self.records[pid]["vector"]), pid))[:limit]

    def graph(self, entities, seeds, hops, limit, query_vector, question, trace):
        scored = []
        for pid, record in self.records.items():
            for rel in record["relations"]:
                if rel.get("embedding"):
                    scored.append((cosine(query_vector, rel["embedding"]), pid, rel))
        scored.sort(key=lambda item: (-item[0], item[1]))
        frontier = set(entities)
        for _, _, rel in scored[:5]:
            frontier.update((rel["subject"], rel["object"]))
        for pid in seeds[:5]:
            for rel in self.records[pid]["relations"]:
                frontier.update((rel["subject"], rel["object"]))
        expanded = []
        for _ in range(hops):
            next_entities = set()
            for item in scored:
                rel = item[2]
                if frontier.intersection((rel["subject"], rel["object"])):
                    if item not in expanded:
                        expanded.append(item)
                    next_entities.update((rel["subject"], rel["object"]))
            frontier = next_entities
        # Expansion participates before relation reranking; direct top K cannot crowd
        # out every bridge. Alternate both branches and retain unique relation IDs.
        pool, seen = [], set()
        from itertools import zip_longest
        for pair in zip_longest(expanded, scored):
            for item in pair:
                if item is None:
                    continue
                key = digest([item[1], item[2]])
                if key not in seen:
                    seen.add(key)
                    pool.append(item)
                if len(pool) >= max(limit, 64):
                    break
            if len(pool) >= max(limit, 64):
                break
        trace.add("graph_expansion", expanded=len(expanded), candidates=len(pool), hops=hops)
        if pool and self.relation_ranker:
            pool = self.relation_ranker(question, pool, trace)
        return list(dict.fromkeys(pid for _, pid, _ in pool))[:limit]

    def collect(self, queries, entities, config, trace):
        ranks, channels = defaultdict(float), defaultdict(list)
        # Queries arrive round-robin by question/slot; no first-slot truncation.
        branches = []
        for slot, question in queries:
            dense = self.dense(question, config.per_channel)
            lexical = self.lexical(question, config.per_channel)
            graph = self.graph(entities, dense + lexical, config.graph_hops, config.per_channel, self.query_vectors[question], question, trace)
            for channel, ids in (("dense", dense), ("bm25", lexical), ("graph", graph)):
                branches.append(ids)
                trace.add("candidates", slot=slot, channel=channel, ids=ids)
                for rank, pid in enumerate(ids):
                    ranks[pid] += 1 / (60 + rank + 1)
                    channels[pid].append({"slot": slot, "channel": channel, "rank": rank + 1})
        # Reserve the first hit of each branch before global fusion.
        ordered = list(dict.fromkeys(ids[0] for ids in branches if ids))
        ordered += [pid for pid in sorted(ranks, key=lambda p: (-ranks[p], p)) if pid not in ordered]
        selected = ordered[:config.candidate_limit]
        trace.add("fusion", ids=selected, provenance={p: channels[p] for p in selected})
        return selected
