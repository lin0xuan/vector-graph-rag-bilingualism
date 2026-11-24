"""Evidence-first Chinese pipeline with bounded retrieval and answer repair."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from decimal import Decimal
import re
from .config import ChineseConfig, PIPELINE_VERSION
from .models import Evidence, Passage, QuestionSlot, Result, Trace, digest, vector
from .retrieval import Retriever


class ChinesePipeline:
    def __init__(self, store, namespace, embedder, model, *, config=None, snapshot=None,
                 mirror=None, count_tokens=len, reranker=None, tokenizer_id="characters-v1", relation_reranker=None,
                 local_search=False):
        if type(local_search) is not bool:
            raise ValueError("local_search must be a boolean")
        if local_search and mirror is not None:
            raise ValueError("local_search cannot be combined with a Milvus mirror")
        if mirror is None and not local_search:
            from .milvus import MilvusMirror
            mirror = MilvusMirror.connect()
        self.store, self.namespace = store, namespace
        self.embedder, self.model = embedder, model
        self.config = config or ChineseConfig()
        self.snapshot, self.mirror, self.count_tokens = snapshot, mirror, count_tokens
        self.reranker = reranker
        self.relation_reranker = relation_reranker
        if count_tokens is not len and tokenizer_id == "characters-v1":
            raise ValueError("Provide tokenizer_id for a custom token counter")
        self.tokenizer_id = tokenizer_id

    def build(self, passages, *, ingestion_config=None, activate=True, resume=True, retry_failed=False):
        trace = Trace()
        relations = {}
        for passage in passages:
            cache_key = digest(["extract-v1", asdict(passage), self.model.identity()])
            cached = self.store.cache_get(cache_key) if resume else None
            if cached and not (retry_failed and cached["status"] == "failed"):
                relations[passage.id] = cached["relations"]
                trace.add("extraction", passage_id=passage.id, status=cached["status"], cached=True, error=cached.get("error"))
                continue
            try:
                response = self.model.call("extract", {"text": passage.text}, trace)
                extracted = response["relations"]
                if not isinstance(extracted, list):
                    raise ValueError("Invalid relations")
                clean = []
                for relation in extracted:
                    if not all(isinstance(relation.get(k), str) and relation[k].strip()
                               for k in ("subject", "predicate", "object", "quote")):
                        raise ValueError("Incomplete relation")
                    if any(relation[k] not in passage.text for k in ("subject", "object", "quote")):
                        raise ValueError("Relation is not grounded in source")
                    clean.append({k: relation[k] for k in ("subject", "predicate", "object", "quote", "time", "condition") if k in relation})
                relations[passage.id] = clean
                trace.add("extraction", passage_id=passage.id, status="success" if clean else "empty")
                self.store.cache_put(cache_key, {"relations": clean, "status": "success" if clean else "empty"})
            except Exception as error:
                relations[passage.id] = []
                trace.add("extraction", passage_id=passage.id, status="failed", error=type(error).__name__)
                self.store.cache_put(cache_key, {"relations": [], "status": "failed", "error": type(error).__name__})
        def embed_cached(texts):
            output = []
            for text in texts:
                key = digest(["embedding-v1", text, self.embedder.identity()])
                cached = self.store.cache_get(key) if resume else None
                if cached is None:
                    values = self.embedder.embed([text])
                    if len(values) != 1:
                        raise ValueError("Embedding response count mismatch")
                    cached = vector(values[0])
                    self.store.cache_put(key, cached)
                output.append(cached)
            return output
        vectors = embed_cached([p.text for p in passages])
        # Embed relations independently so graph retrieval is not tied to lexical entity matches.
        flat = [rel for group in relations.values() for rel in group]
        if flat:
            rel_vectors = embed_cached([f'{r["subject"]} {r["predicate"]} {r["object"]}' for r in flat])
            if len(rel_vectors) != len(flat):
                raise ValueError("Relation embedding count mismatch")
            for relation, values in zip(flat, rel_vectors):
                relation["embedding"] = vector(values, len(vectors[0]))
        index_config = {"pipeline_schema": 1, "embedding": self.embedder.identity(),
                        "extraction": self.model.identity(), "ingestion": ingestion_config or {}, "tokenizer": self.tokenizer_id}
        snapshot = self.store.create(self.namespace, "zh", passages, vectors, relations, index_config,
                                     telemetry=trace.events)
        manifest, records = self.store.load(snapshot)
        if self.mirror:
            self.mirror.ensure(snapshot, manifest, records)
        if activate:
            self.store.activate(self.namespace, snapshot)
        return snapshot

    def sync_milvus(self):
        """Copy an existing snapshot without extraction or embedding calls."""
        if self.mirror is None:
            raise ValueError("sync_milvus requires Standalone storage")
        snapshot = self.snapshot or self.store.current(self.namespace)
        manifest, records = self.store.load(snapshot)
        if manifest["namespace"] != self.namespace or manifest["language"] != "zh":
            raise ValueError("Snapshot namespace/language mismatch")
        self.mirror.ensure(snapshot, manifest, records)
        return snapshot

    def _load(self):
        snapshot = self.snapshot or self.store.current(self.namespace)
        manifest, records = self.store.load(snapshot)
        if manifest["namespace"] != self.namespace or manifest["language"] != "zh":
            raise ValueError("Snapshot namespace/language mismatch")
        if manifest["config"]["embedding"] != self.embedder.identity():
            raise ValueError("Embedding configuration differs from index")
        if self.mirror:
            self.mirror.validate(snapshot, records)
        return snapshot, records

    def fingerprint(self):
        return digest({"pipeline": PIPELINE_VERSION, "config": asdict(self.config),
                       "storage": "milvus-standalone" if self.mirror else "local-exact",
                       "models": self.model.identity(), "embedding": self.embedder.identity(),
                       "reranker": self.reranker.identity() if self.reranker else None,
                       "relation_reranker": self.relation_reranker.identity() if self.relation_reranker else None,
                       "tokenizer": self.tokenizer_id})

    def _plan(self, question, trace):
        try:
            items = self.model.call("plan", {"question": question, "max_slots": self.config.max_slots}, trace)["slots"]
            if not isinstance(items, list) or not 1 <= len(items) <= self.config.max_slots:
                raise ValueError("Invalid question plan")
            slots = [QuestionSlot(**item) for item in items]
            if any(not isinstance(s.id, str) or not isinstance(s.question, str) or not s.id.strip() or not s.question.strip() for s in slots):
                raise ValueError("Empty slot")
            if len({s.id for s in slots}) != len(slots):
                raise ValueError("Duplicate slot IDs")
            trace.add("plan", status="success", slots=[asdict(s) for s in slots])
        except Exception as error:
            slots = [QuestionSlot("s1", question)]
            trace.add("plan", status="fallback", error=type(error).__name__, slots=[asdict(s) for s in slots])
        return slots

    def _entities(self, question, trace):
        try:
            entities = self.model.call("entities", {"question": question}, trace)["entities"]
            if not isinstance(entities, list) or any(not isinstance(e, str) or not e or e not in question for e in entities):
                raise ValueError("Invalid query entities")
            trace.add("entities", status="success" if entities else "empty", entities=entities)
            return entities
        except Exception as error:
            trace.add("entities", status="failed", error=type(error).__name__)
            return []

    def _rank_relations(self, question, scored, trace):
        if self.config.retrieval_mode == "single-stage":
            return scored
        # Model-facing IDs are local aliases, never hashes that the model must copy.
        keyed = {f"r{i + 1}": row for i, row in enumerate(scored)}
        try:
            if self.relation_reranker:
                texts = [f'{rel["subject"]} {rel["predicate"]} {rel["object"]}' for _, _, rel in scored]
                order = self.relation_reranker.rank_relations(question, texts)
                if sorted(order) != list(range(len(scored))):
                    raise ValueError("Invalid relation permutation")
                trace.add("relation_rerank", status="success", backend=self.relation_reranker.identity(),
                          selected=[scored[i][1] for i in order])
                return [scored[i] for i in order]
            ids = self.model.call("relations", {"question": question, "relations": [
                {"id": key, **{k: rel[k] for k in ("subject", "predicate", "object", "time", "condition") if k in rel}}
                for key, (_, pid, rel) in keyed.items()]}, trace)["ids"]
            if not isinstance(ids, list) or any(not isinstance(i, str) or i not in keyed for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("Invalid relation selection")
            trace.add("relation_rerank", status="success", selected=[keyed[i][1] for i in ids])
            return [keyed[i] for i in ids]
        except Exception as error:
            trace.add("relation_rerank", status="fallback", error=type(error).__name__)
            return scored

    @staticmethod
    def _sentences(text):
        # Offsets reference the untouched original, including punctuation/newlines.
        return [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^。！？\n]+[。！？\n]*|[。！？\n]+", text)]

    def _select(self, question, slots, ids, passages, trace):
        if ids and self.reranker:
            try:
                order = self.reranker.rank(question, [passages[pid].text for pid in ids])
                if sorted(order) != list(range(len(ids))):
                    raise ValueError("Invalid passage permutation")
                ids = [ids[i] for i in order]
                trace.add("passage_rerank", status="success", ids=ids)
            except Exception as error:
                trace.add("passage_rerank", status="fallback", error=type(error).__name__)

        evidence = []
        slot_ids = {s.id for s in slots}
        batches = [ids[start:start + self.config.selection_batch]
                   for start in range(0, len(ids), self.config.selection_batch)]

        def select_batch(batch):
            aliases = {f"p{i + 1}": pid for i, pid in enumerate(batch)}
            sentences = {alias: self._sentences(passages[pid].text) for alias, pid in aliases.items()}
            try:
                response = self.model.call("select", {"question": question, "slots": [asdict(s) for s in slots],
                    "candidates": [{"id": alias, "sentences": [
                        {"id": i + 1, "text": row[2]} for i, row in enumerate(sentences[alias])
                    ]} for alias in aliases]}, trace)
                items = response["evidence"]
                if not isinstance(items, list):
                    raise ValueError("Invalid selection")
                valid, rejected = [], 0
                for item in items:
                    try:
                        alias, slot, indexes = item["passage_id"], item["slot_id"], item["sentence_ids"]
                        if alias not in aliases or slot not in slot_ids or not isinstance(indexes, list) or not indexes:
                            raise ValueError("Unknown evidence reference")
                        if any(type(i) is not int or not 1 <= i <= len(sentences[alias]) for i in indexes):
                            raise ValueError("Unknown sentence")
                        for i in dict.fromkeys(indexes):
                            begin, end, quote = sentences[alias][i - 1]
                            ev = Evidence(aliases[alias], slot, quote, begin, end)
                            if ev not in valid:
                                valid.append(ev)
                    except (KeyError, TypeError, ValueError):
                        rejected += 1
                return batch, valid, rejected, None
            except Exception as error:
                return batch, [], 0, error

        if len(batches) > 1:
            # Batches do not depend on each other. Keep merge order so evidence
            # and failure events stay identical to the previous serial pass.
            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
                selected_batches = list(pool.map(select_batch, batches))
        else:
            selected_batches = [select_batch(batch) for batch in batches]
        for batch, valid, rejected, error in selected_batches:
            if error is not None:
                trace.add("selection", status="failed", error=type(error).__name__, ids=batch)
                continue
            evidence.extend(e for e in valid if e not in evidence)
            for _ in range(rejected):
                trace.add("rejected_evidence", reason="unknown_id_or_sentence")
            if rejected:
                # Invalid references are a technical failure, never confirmed coverage.
                trace.add("selection", status="failed", error="InvalidEvidence", ids=batch)
        trace.add("selection", evidence=[asdict(e) for e in evidence])
        return evidence

    def _pack(self, slots, evidence, passages, trace, ranked_ids=None):
        by_slot = {s.id: list(dict.fromkeys(e.passage_id for e in evidence if e.slot_id == s.id)) for s in slots}
        selected, used = [], 0
        def add(pid):
            nonlocal used
            if pid in selected:
                return
            cost = self.count_tokens(passages[pid].text)
            if used + cost <= self.config.context_budget:
                selected.append(pid)
                used += cost
            else:
                trace.add("context_budget", skipped=pid)
        for rank in range(max((len(ids) for ids in by_slot.values()), default=0)):
            for ids in by_slot.values():
                if rank < len(ids):
                    add(ids[rank])
        # A failed selector cannot erase its input. Preserve ranked originals as
        # explicitly unconfirmed context; do not invent quotes or mark slots covered.
        failed_batches = [e["ids"] for e in trace.events if e["stage"] == "selection" and e.get("status") == "failed"]
        fallback = []
        for rank in range(max(map(len, failed_batches), default=0)):
            for batch in failed_batches:
                if rank < len(batch) and batch[rank] not in selected:
                    pid = batch[rank]
                    add(pid)
                    if pid in selected:
                        fallback.append(pid)
        if fallback:
            trace.add("selection_fallback", ids=fallback, confirmed=False)
        # Keep high-ranked full passages available even when sentence selection
        # misses a fact. Sentence selection annotates evidence; it does not
        # decide whether a retrieved passage exists in the answer context.
        for pid in (ranked_ids or []):
            add(pid)
        packed = [e for e in evidence if e.passage_id in selected]
        missing = [s.id for s in slots if not any(e.slot_id == s.id for e in packed)]
        trace.add("coverage", selected=selected, missing=missing, context_units=used)
        return selected, packed, missing

    def _repair(self, question, slots, gaps, state, trace, *, reason):
        if state["rounds"] >= self.config.max_repairs or not gaps:
            return False
        state["rounds"] += 1
        queries = [(g["slot_id"], g["query"]) for g in gaps]
        found = state["retriever"].collect(queries, state["entities"], self.config, trace)
        passages = state["passages"]
        for pid in state["selected"] or state["ids"][:5]:
            p = passages[pid]
            found.extend(n for n in p.neighbors if n in passages and passages[n].document_id == p.document_id)
            found.extend(other.id for other in passages.values()
                         if p.parent_id and other.parent_id == p.parent_id and other.document_id == p.document_id)
        # Revisit previously rejected candidates under the specific gap query. Seeing
        # a passage in an earlier pool is not equivalent to selecting its facts.
        candidates = list(dict.fromkeys(found))[:self.config.candidate_limit]
        added = [pid for pid in candidates if pid not in state["seen"]]
        trace.add("repair", round=state["rounds"], reason=reason, missing=[g["slot_id"] for g in gaps],
                  queries=queries, added=added, reconsidered=[pid for pid in candidates if pid in state["seen"]])
        state["seen"].update(candidates)
        if not candidates:
            return False
        targets = [QuestionSlot(g["slot_id"], g["query"]) for g in gaps]
        extra = self._select(question, targets, candidates, passages, trace)
        previous = set((e.passage_id, e.slot_id, e.start, e.end) for e in state["evidence"])
        state["evidence"].extend(e for e in extra if e not in state["evidence"])
        state["selected"], state["packed"], state["missing"] = self._pack(
            slots, state["evidence"], passages, trace, ranked_ids=candidates)
        return any((e.passage_id, e.slot_id, e.start, e.end) not in previous for e in extra)

    def _retrieve(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question is required")
        snapshot, records = self._load()  # Strict validation precedes all model requests.
        trace = Trace()
        manifest, _ = self.store.load(snapshot)
        failures = [e for e in manifest["telemetry"] if e.get("stage") == "extraction" and e.get("status") == "failed"]
        trace.add("index_health", extraction_failures=len(failures), failed_passages=[e["passage_id"] for e in failures])
        slots = self._plan(question, trace)
        entities = self._entities(question, trace)
        passages = {r["passage"]["id"]: Passage(**r["passage"]) for r in records}
        retriever = Retriever(records, self.embedder, mirror=self.mirror, snapshot=snapshot, relation_ranker=self._rank_relations)
        ids = retriever.collect([("original", question)] + [(s.id, s.question) for s in slots], entities, self.config, trace)
        evidence = self._select(question, slots, ids, passages, trace)
        all_evidence = evidence
        selected, evidence, missing = self._pack(slots, evidence, passages, trace, ranked_ids=ids)
        state = {"retriever": retriever, "passages": passages, "entities": entities, "ids": ids,
                 "seen": set(ids), "rounds": 0, "evidence": all_evidence, "selected": selected,
                 "packed": evidence, "missing": missing}
        while state["missing"] and state["rounds"] < self.config.max_repairs:
            gaps = [{"slot_id": s.id, "query": s.question} for s in slots if s.id in state["missing"]]
            if not self._repair(question, slots, gaps, state, trace, reason="selection_gap"):
                break
        selected, evidence, missing = state["selected"], state["packed"], state["missing"]
        fallback = any(e["stage"] == "relation_rerank" and e.get("status") == "fallback" for e in trace.events)
        trace.add("execution", requested_mode=self.config.retrieval_mode,
                  effective_mode="single-stage" if fallback else self.config.retrieval_mode,
                  fallback_reason="relation_rerank_failed" if fallback else None)
        degraded = any(e.get("status") in ("failed", "fallback") for e in trace.events)
        result = Result(question, "zh", self.config.profile, snapshot,
                        "partial" if missing or degraded else "complete",
                        passages=[passages[pid] for pid in selected], evidence=evidence,
                        missing_slots=missing, trace=trace)
        return result, slots, state

    def retrieve(self, question):
        return self._retrieve(question)[0]

    @staticmethod
    def _normalize_citations(answer):
        # Support [p1,p2] and [p1，p2] without fuzzy ID repair.
        return re.sub(r"\[([^\[\]]+)\]", lambda m: "".join(
            f"[{part.strip()}]" for part in re.split(r"[,，;；]", m[1]) if part.strip()), answer)

    def _verify(self, answer, payload, trace):
        answer = self._normalize_citations(answer)
        citations = re.findall(r"\[([^\[\]]+)\]", answer)
        by_id = {p["id"]: p["text"] for p in payload["passages"]}
        issues = []
        # A verifier must not turn an unconfirmed retrieval slot into a
        # complete answer merely because the prose looks plausible.  The
        # selector is the component that establishes which source sentence
        # supports each slot; keep unresolved slots visible until a repair
        # finds evidence for them.
        evidence_slots = {item.get("slot_id") for item in payload.get("evidence", [])
                          if isinstance(item, dict) and isinstance(item.get("slot_id"), str)}
        slot_questions = {item.get("id"): item.get("question", "")
                          for item in payload.get("slots", []) if isinstance(item, dict)}
        unresolved = [slot_id for slot_id in payload.get("missing_slots", [])
                      if slot_id not in evidence_slots]
        for slot_id in unresolved:
            issues.append("unconfirmed_slot:" + slot_id)
        def numbers(text):
            values = set()
            for n in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", text):
                percent = "%" if n.endswith("%") else ""
                value = format(Decimal(n.rstrip("%").replace(",", "")), "f")
                if "." in value:
                    value = value.rstrip("0").rstrip(".")
                values.add(value + percent)
            return values
        for line in answer.splitlines():
            refs = re.findall(r"\[([^\[\]]+)\]", line)
            if refs and all(ref in by_id for ref in refs):
                literal = re.sub(r"\[[^\[\]]+\]", "", line).replace("**", "")
                # Strip actual local aliases even when prose says “原文 p1”.
                for alias in [*by_id, *(slot["id"] for slot in payload.get("slots", []))]:
                    literal = re.sub(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", "", literal)
                # Detect inline sequences BEFORE stripping the first line marker.
                # Otherwise a two-item line loses 1 first and misclassifies lone 2.
                markers = list(re.finditer(r"(?<![A-Za-z0-9_.])(?:[（(](\d{1,3})[)）]|(\d{1,3})[.、)）](?!\d))\s*", literal))
                ordinals = [int(m[1] or m[2]) for m in markers]
                if len(ordinals) >= 2 and ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals))):
                    for marker in reversed(markers):
                        literal = literal[:marker.start()] + literal[marker.end():]
                literal = re.sub(r"^\s*(?:[-*+]\s+|#{1,6}\s+)?(?:\d+[.)、](?!\d)|[（(]\d+[)）])\s*", "", literal)
                unsupported = numbers(literal) - numbers(" ".join(by_id[ref] for ref in refs))
                if unsupported:
                    issues.append("numeric_copy_mismatch:" + ",".join(sorted(unsupported)))
        if not citations or any(pid not in by_id for pid in citations):
            issues.append("missing_or_unknown_citation")
        checked = self.model.call("verify", {**payload, "answer": answer, "rule_issues": issues}, trace)
        if type(checked.get("ok")) is not bool or not isinstance(checked.get("issues"), list) or any(not isinstance(i, str) for i in checked["issues"]):
            raise ValueError("Invalid verifier response")
        gaps = checked.get("gaps", [])
        valid_slots = {s["id"] for s in payload.get("slots", [])}
        if not isinstance(gaps, list) or any(not isinstance(g, dict) or g.get("slot_id") not in valid_slots
                or not isinstance(g.get("query"), str) or not g["query"].strip() for g in gaps):
            raise ValueError("Invalid verifier gaps")
        gaps = list({g["slot_id"]: {"slot_id": g["slot_id"], "query": g["query"]} for g in gaps}.values())
        # If the model verifier overlooked an unresolved slot, turn it into a
        # concrete repair request.  The query stays tied to the original slot
        # wording so the repair retriever does not invent a target.
        known_gaps = {g["slot_id"] for g in gaps}
        for slot_id in unresolved:
            if slot_id not in known_gaps:
                gaps.append({"slot_id": slot_id, "query": slot_questions.get(slot_id) or slot_id})
        if (not checked["ok"] or gaps) and not checked["issues"]:
            checked["issues"] = ["verifier_rejected_answer"]
        issues += checked["issues"]
        trace.add("verification", ok=not issues and checked["ok"], issues=issues, gaps=gaps)
        return issues

    def _answer_payload(self, result, slots):
        aliases = {p.id: f"p{i + 1}" for i, p in enumerate(result.passages)}
        result.trace.add("citation_map", aliases={v: k for k, v in aliases.items()})
        return {"question": result.question, "slots": [asdict(s) for s in slots],
                "evidence": [{**asdict(e), "passage_id": aliases[e.passage_id]} for e in result.evidence],
                "missing_slots": result.missing_slots,
                "passages": [{"id": aliases[p.id], "text": p.text} for p in result.passages]}, aliases

    @staticmethod
    def _public_answer(answer, aliases):
        reverse = {v: k for k, v in aliases.items()}
        return re.sub(r"\[([^\[\]]+)\]", lambda m: "[" + reverse.get(m[1], m[1]) + "]", answer)

    def query(self, question):
        result, slots, state = self._retrieve(question)
        if not result.passages:
            result.status = "partial"
            result.answer = "当前证据不足，无法可靠回答这些问点。"
            return result
        payload, aliases = self._answer_payload(result, slots)
        rewrite_used = False
        last_answer = None
        while True:
            try:
                answer = self.model.call("answer", payload, result.trace)["answer"]
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("Empty answer")
                answer = self._normalize_citations(answer)
                result.answer = self._public_answer(answer, aliases)
                last_answer = {"answer": result.answer, "passages": list(result.passages),
                               "evidence": list(result.evidence), "missing_slots": list(result.missing_slots)}
                result.trace.add("answer_attempt", answer=result.answer, passage_ids=[p.id for p in result.passages])
                issues = self._verify(answer, payload, result.trace)
                gaps = result.trace.events[-1].get("gaps", [])
                if not issues:
                    # An accepted verifier can confirm context preserved after a
                    # selector failure, but technical degradation remains visible.
                    result.missing_slots = []
                    degraded = any(e.get("status") in ("failed", "fallback") for e in result.trace.events)
                    result.status = "partial" if degraded else "complete"
                    break
                if gaps:
                    result.missing_slots = list(dict.fromkeys(result.missing_slots + [g["slot_id"] for g in gaps]))
                    if state["rounds"] < self.config.max_repairs:
                        before = list(state["selected"])
                        changed = self._repair(question, slots, gaps, state, result.trace, reason="verification_gap")
                        result.passages = [state["passages"][pid] for pid in state["selected"]]
                        result.evidence = state["packed"]
                        if changed or before != state["selected"]:
                            # Rebuild aliases and generate afresh: old aliases must
                            # never point to newly ordered passages.
                            result.answer = ""
                            payload, aliases = self._answer_payload(result, slots)
                            payload["issues"] = issues
                            continue
                if not rewrite_used:
                    rewrite_used = True
                    result.trace.add("answer_repair", original=result.answer, issues=issues)
                    payload = {**payload, "previous_answer": answer, "issues": issues}
                    continue
                result.status = "partial"
                break
            except Exception as error:
                result.trace.add("generation", status="failed", error=type(error).__name__)
                if last_answer:
                    result.answer = last_answer["answer"]
                    result.passages = last_answer["passages"]
                    result.evidence = last_answer["evidence"]
                    result.missing_slots = list(dict.fromkeys(last_answer["missing_slots"] + result.missing_slots))
                    result.trace.add("answer_fallback", status="fallback", reason=type(error).__name__,
                                     retained_passages=[p.id for p in result.passages])
                result.status = "partial" if result.answer else "failed"
                break
        return result
