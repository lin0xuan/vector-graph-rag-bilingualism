"""Pinned upstream indexing/retrieval with optional Jev or generative passage ranking."""
from dataclasses import asdict
from importlib.metadata import distribution
import json
import os
import subprocess
from pathlib import Path
from .config import EnglishConfig, EnglishQueryConfig, UPSTREAM_COMMIT
from .english_reranking import GenerativeRanker, VERSION, interleave
from .milvus import collection_hashes, dump_collection
from .models import Passage, Result, Trace, digest
from .storage import IntegrityError


def verify_upstream_version():
    dist = distribution("vector-graph-rag")
    info = json.loads(dist.read_text("direct_url.json") or "{}")
    if info.get("vcs_info", {}).get("commit_id") == UPSTREAM_COMMIT:
        return
    # Editable source check supports local offline integration of the pinned checkout.
    if info.get("dir_info", {}).get("editable") and info.get("url", "").startswith("file://"):
        from urllib.parse import unquote, urlsplit
        path = unquote(urlsplit(info["url"]).path)
        head = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if head == UPSTREAM_COMMIT and not dirty:
            return
    raise ValueError("Install the English extra at the pinned upstream commit")


def safe_settings(settings):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if not any(secret in k.lower() for secret in ("api_key", "token", "password"))}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(settings.model_dump(mode="json"))


class EnglishUpstreamPipeline:
    def __init__(self, upstream, store, namespace, *, snapshot=None, profile="en_upstream_jev_1_13", query_config=None):
        self.upstream, self.store, self.namespace = upstream, store, namespace
        self.snapshot, self.profile = snapshot, profile
        self.query_config = query_config or EnglishQueryConfig()
        # Isolate all version-sensitive private access in this adapter.
        storage = upstream._store
        self.client = storage.client
        self.collections = [storage.entity_collection, storage.relation_collection, storage.passage_collection]
        self.passage_collection = storage.passage_collection
        self.settings = safe_settings(upstream.settings)
        self.identity = {"commit": UPSTREAM_COMMIT, "settings": self.settings}

    @classmethod
    def connect(cls, store, namespace, config, *, snapshot=None, reranker=None):
        verify_upstream_version()
        from vector_graph_rag import VectorGraphRAG
        from vector_graph_rag.config import Settings
        required = {"embedding_provider", "embedding_model", "embedding_dimension", "llm_model", "final_top_k", "collection_prefix"}
        if not required.issubset(config.settings):
            raise ValueError(f"English profile must explicitly set {sorted(required)}")
        # The default English path uses generative relation/passage ranking. Jev is an
        # explicit opt-in and must never be enabled merely by constructing the
        # adapter.
        kwargs = {k: v for k, v in config.settings.items() if k != "reranker_model"}
        # Share the Chinese SiliconFlow-compatible credentials when present.
        # The profile contains only the endpoint; the secret stays in .env.
        if os.getenv("VGRAG_CHAT_KEY") and "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("VGRAG_CHAT_KEY")
        accounting = None
        jev_fallback = None
        if reranker is not None:
            try:
                from .jev_service import resolve_service, model_connection, JevAccounting
                options = dict(reranker)
                if options.pop("provider", "jev") != "jev":
                    raise ValueError("English pinned profile requires Jev")
                options = resolve_service(**options)
                if options["model"] != "jev-1.13.0":
                    raise ValueError("This English profile pins jev-1.13.0")
                kwargs.update(reranker_model=model_connection(options),
                              jev_timeout=options["timeout"], jev_max_concurrency=options["max_concurrency"])
                accounting = JevAccounting(options)
            except Exception as error:
                jev_fallback = type(error).__name__
        settings = Settings(**kwargs)
        upstream = VectorGraphRAG(settings=settings)
        if accounting is not None:
            accounting.attach(upstream._reranker)
        result = cls(upstream, store, namespace, snapshot=snapshot, profile=config.profile,
                     query_config=config.query)
        result.jev_accounting = accounting
        result.jev_fallback = jev_fallback
        return result

    def build(self, passages, *, extract_triplets=True, triplets=None):
        from vector_graph_rag.models import Document
        if any(dump_collection(self.client, name) for name in self.collections):
            raise ValueError("English build requires a fresh collection_prefix; refusing to replace existing data")
        if not passages or any(p.language != "en" for p in passages):
            raise ValueError("English documents required")
        if len({p.id for p in passages}) != len(passages):
            raise ValueError("Duplicate passage IDs")
        docs = [Document(id=p.id, page_content=p.text,
                         metadata={**p.metadata, "source": p.source,
                                   "triplets": (triplets or {}).get(p.id, [])}) for p in passages]
        self.upstream.rebuild_documents(docs, extract_triplets=extract_triplets, show_progress=False)
        rows = {row["id"]: row for row in dump_collection(self.client, self.passage_collection)}
        if set(rows) != {p.id for p in passages}:
            raise IntegrityError("Upstream passage identities differ from supplied documents")
        external = {"collections": collection_hashes(self.client, self.collections)}
        snapshot = self.store.create(self.namespace, "en", passages, [rows[p.id]["vector"] for p in passages],
                                     {}, self.identity, external=external)
        self.store.activate(self.namespace, snapshot)
        return snapshot

    def _validate(self):
        snapshot = self.snapshot or self.store.current(self.namespace)
        manifest, records = self.store.load(snapshot)
        if manifest["namespace"] != self.namespace or manifest["language"] != "en":
            raise ValueError("English namespace/language mismatch")
        if manifest["config"] != self.identity:
            raise ValueError("English upstream configuration changed; use a matching snapshot")
        expected = manifest["external"]["collections"]
        if set(expected) != set(self.collections) or collection_hashes(self.client, self.collections) != expected:
            raise IntegrityError("Upstream Milvus content changed")
        return snapshot, [Passage(**r["passage"]) for r in records]

    def _run(self, question, method, *, filter=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question is required")
        snapshot, passages = self._validate()
        if filter:
            allowed = {row["id"] for row in self.upstream._store.get_passages_by_ids(
                [p.id for p in passages], filter=filter)}
            passages = [p for p in passages if p.id in allowed]
        # Forward filter unchanged to the native API; never post-filter top K.
        ranking_events = []
        try:
            if not self.upstream.settings.uses_jev:
                raw, ranking_events = self._run_generative(question, method, passages, filter=filter)
            else:
                raw = getattr(self.upstream, method)(question, filter=filter)
        except Exception as error:
            if not self.upstream.settings.uses_jev:
                raise
            raw = (self.upstream.query_naive(question, filter=filter)
                   if method == "query"
                   else self.upstream.retrieve(question, filter=filter, use_reranking=False))
            trace_fallback = type(error).__name__
        else:
            trace_fallback = None
        texts = raw.passages or raw.retrieved_passages
        trace = Trace()
        if trace_fallback:
            trace.add("reranker", provider="jev", status="fallback", error=trace_fallback)
        if getattr(self, "jev_fallback", None):
            trace.add("reranker", provider="jev", status="fallback", error=self.jev_fallback)
        mapped, ambiguous, unmapped = [], [], []
        for text in texts:
            matches = [p for p in passages if p.text == text]
            if len(matches) > 1:
                ambiguous.append({"text_hash": digest(text), "source_aliases": [p.source for p in matches]})
            if not matches:
                unmapped.append(digest(text))
            for p in matches:
                if p not in mapped:
                    mapped.append(p)
        trace.add("upstream", method=method, commit=UPSTREAM_COMMIT, passage_texts=texts,
                  ambiguous_citations=ambiguous, unmapped_texts=unmapped,
                  query_entities=raw.query_entities, retrieved_relations=raw.retrieved_relations,
                  expanded_relations=raw.expanded_relations, reranked_relations=raw.reranked_relations)
        for event in ranking_events:
            trace.add(**event)
        # Status describes execution/provenance, not an assertion that the answer is correct.
        return Result(question, "en", self.profile, snapshot,
                      "partial" if ambiguous or unmapped or not texts else "complete",
                      answer=raw.answer, passages=mapped, trace=trace)

    def _run_generative(self, question, method, passage_records, *, filter=None):
        """Share the same evidence ranking between query and retrieval-only evals."""
        from vector_graph_rag.models import QueryResult, RetrievalDetail, RerankResult, EvictionResult

        upstream = self.upstream
        top_k = upstream.settings.final_top_k
        config = self.query_config
        ranker = GenerativeRanker(upstream._reranker.client, upstream._reranker.model, config)
        with upstream._observed_operation(f"vgrag.{method}", {
            "vgrag.use_reranking": True, "vgrag.compare_naive": False,
            "vgrag.has_filter": bool(filter), "vgrag.question_length": len(question),
        }):
            retriever = upstream._ensure_retriever()
            retrieved = retriever.retrieve(question, filter=filter)
            # Prefer directly retrieved relations in a failure fallback, rather
            # than the upstream expanded table's arbitrary UUID ordering.
            by_id = dict(zip(retrieved.expanded_relation_ids, retrieved.expanded_relation_texts))
            candidate_ids = list(dict.fromkeys(
                [rid for rid in retrieved.relation_ids if rid in by_id] + list(by_id)))
            candidate_texts = [by_id[rid] for rid in candidate_ids]
            order, events = ranker.rank(question, candidate_texts, stage="relation_rerank",
                                         limit=config.relation_limit)
            selected_ids = [candidate_ids[i] for i in order]
            selected_texts = [candidate_texts[i] for i in order]
            rerank_result = RerankResult(selected_relation_ids=selected_ids,
                                         selected_relation_texts=selected_texts)
            _, graph = upstream._get_passages_from_relations(selected_ids, filter=filter, round_robin=True)
            dense = retriever.retrieve_passages_naive(question, top_k=max(config.dense_top_k, top_k), filter=filter)
            candidates = interleave(graph, dense, max(config.passage_limit, top_k))
            sources = {}
            for p in passage_records:
                sources.setdefault(p.text, []).append({"source": p.source, "title": p.metadata.get("title", "")})
            rank_texts = [json.dumps({"sources": sources.get(text, []), "text": text}, ensure_ascii=False)
                          for text in candidates]
            order, passage_events = ranker.rank(question, rank_texts, stage="passage_rerank",
                                                 limit=len(candidates))
            final_passages = [candidates[i] for i in order[:top_k]]
            events += passage_events
            events.append({"stage": "english_retrieval", "version": VERSION,
                           "query_config": asdict(config), "graph_passages": len(graph),
                           "dense_passages": len(dense), "candidate_passages": candidates,
                           "passage_ranking_inputs": rank_texts,
                           "relation_candidates": candidate_texts, "final_passages": len(final_passages)})
            answer = upstream._answer_generator.generate(question, final_passages) if method == "query" else ""
            return QueryResult(
                query=question, answer=answer, query_entities=retrieved.query_entities,
                passages=final_passages, retrieved_passages=final_passages,
                retrieved_relations=retrieved.relation_texts, expanded_relations=candidate_texts,
                reranked_relations=selected_texts, subgraph=retrieved.subgraph,
                retrieval_detail=RetrievalDetail(
                    entity_ids=retrieved.entity_ids, entity_texts=retrieved.entity_texts,
                    entity_scores=retrieved.entity_scores, relation_ids=retrieved.relation_ids,
                    relation_texts=retrieved.relation_texts, relation_scores=retrieved.relation_scores),
                rerank_result=rerank_result,
                eviction_result=EvictionResult(occurred=retrieved.eviction_occurred,
                    before_count=retrieved.eviction_before_count, after_count=retrieved.eviction_after_count),
            ), events

    def query(self, question, *, filter=None):
        return self._run(question, "query", filter=filter)

    def retrieve(self, question, *, filter=None):
        return self._run(question, "retrieve", filter=filter)

    def fingerprint(self):
        # Query policy changes must not reuse evaluation results, but they do
        # not change the immutable index identity or require re-extraction.
        return digest({"identity": self.identity, "profile": self.profile,
                       "retrieval": "upstream_jev" if self.upstream.settings.uses_jev else VERSION,
                       "query": asdict(self.query_config)})
