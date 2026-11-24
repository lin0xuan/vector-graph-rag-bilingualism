"""Chinese bridge to the pinned upstream Jev /systemone scoring protocol.

Unlike a generic /rerank endpoint, Jev returns per-candidate noul scores against
shared context. English continues to use the unmodified native upstream path.
"""
from .config import UPSTREAM_COMMIT
from .english import verify_upstream_version


class JevReranker:
    def __init__(self, *, model="jev-1.13.0", key_env=None, base_url=None,
                 timeout=60, max_concurrency=None, service="typesafe", request_observer=None):
        from .jev_service import resolve_service, model_connection, JevAccounting
        options = resolve_service(service=service, model=model, key_env=key_env,
                                  base_url=base_url, timeout=timeout, max_concurrency=max_concurrency)
        connection = model_connection(options)
        verify_upstream_version()
        from vector_graph_rag.config import Settings
        from vector_graph_rag.llm.jev import JevReranker as NativeJev
        settings = Settings(reranker_model=connection, use_llm_cache=False,
                            jev_timeout=timeout, jev_max_concurrency=options["max_concurrency"])
        self.native = NativeJev(settings)
        self.accounting = JevAccounting(options, request_observer).attach(self.native)
        self._identity = {"provider": "jev", "service": service, "model": model, "base_url": options["base_url"],
                          "upstream_commit": UPSTREAM_COMMIT, "passage_batch": 16,
                          "protocol": "systemone-noul", "recipe": "zh-all-candidates-v1"}

    def identity(self):
        return dict(self._identity)

    def rank_relations(self, question, texts):
        ids = [str(i) for i in range(len(texts))]
        scores = self.native.score_relations(question, ids, texts)
        return sorted(range(len(texts)), key=lambda i: -scores[str(i)])

    def rank(self, question, texts):
        batches = []
        def split(indices):
            try:
                payload = self.native.build_passage_request(question, [texts[i] for i in indices])
            except ValueError:
                if len(indices) < 2:
                    raise  # Never silently truncate a single oversized excerpt.
                middle = len(indices) // 2
                split(indices[:middle])
                split(indices[middle:])
                return
            batches.append((indices, payload))
        for start in range(0, len(texts), 16):
            split(list(range(start, min(start + 16, len(texts)))))
        if not batches:
            return []

        def score_batch(item):
            indices, payload = item
            # One payload per call: excerpt ids restart at d0 in every batch.
            values = self.native._score_payloads([payload])
            return {index: values[f"d{i}"] for i, index in enumerate(indices)}

        # The attached accounting semaphore already caps JevHub at its preset.
        workers = min(len(batches), self.native.settings.jev_max_concurrency)
        if workers == 1:
            parts = [score_batch(item) for item in batches]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(score_batch, batches))
        scores = {}
        for part in parts:
            scores.update(part)
        return sorted(range(len(texts)), key=lambda i: -scores[i])
