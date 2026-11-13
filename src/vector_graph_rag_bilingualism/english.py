"""Thin adapter over the pinned upstream; ranking and generation remain upstream."""
from dataclasses import asdict
from importlib.metadata import distribution
import json
import subprocess
from pathlib import Path
from .config import EnglishConfig, UPSTREAM_COMMIT
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
    def __init__(self, upstream, store, namespace, *, snapshot=None, profile="en_upstream_jev_1_13"):
        self.upstream, self.store, self.namespace = upstream, store, namespace
        self.snapshot, self.profile = snapshot, profile
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
        kwargs = {**config.settings, "reranker_model": "jev-1.13.0"}
        accounting = None
        if reranker is not None:
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
        settings = Settings(**kwargs)
        upstream = VectorGraphRAG(settings=settings)
        if accounting is not None:
            accounting.attach(upstream._reranker)
        result = cls(upstream, store, namespace, snapshot=snapshot, profile=config.profile)
        result.jev_accounting = accounting
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
        # Forward filter unchanged to the native API; never post-filter top K.
        raw = getattr(self.upstream, method)(question, filter=filter)
        if filter:
            allowed = {row["id"] for row in self.upstream._store.get_passages_by_ids(
                [p.id for p in passages], filter=filter)}
            passages = [p for p in passages if p.id in allowed]
        texts = raw.passages or raw.retrieved_passages
        trace = Trace()
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
        # Status describes execution/provenance, not an assertion that the answer is correct.
        return Result(question, "en", self.profile, snapshot,
                      "partial" if ambiguous or unmapped or not texts else "complete",
                      answer=raw.answer, passages=mapped, trace=trace)

    def query(self, question, *, filter=None):
        return self._run(question, "query", filter=filter)

    def retrieve(self, question, *, filter=None):
        return self._run(question, "retrieve", filter=filter)

    def fingerprint(self):
        return digest({"identity": self.identity, "profile": self.profile})
