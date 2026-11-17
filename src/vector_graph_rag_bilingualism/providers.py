"""Explicit remote adapters. Construction never makes a request."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .models import canonical, vector


class ProviderError(RuntimeError):
    pass


def post(service, route, body):
    key = os.getenv(service.key_env)
    if not key:
        raise ProviderError(f"Missing credential environment variable: {service.key_env}")
    request = Request(service.base_url.rstrip("/") + route, canonical(body).encode(),
                      {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=service.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            retry = error.code in (429, 500, 502, 503, 504)
            if not retry or attempt == 2:
                # Do not expose response bodies, which may contain credentials/provider internals.
                raise ProviderError(f"Model service HTTP {error.code}") from None
        except (URLError, TimeoutError):
            if attempt == 2:
                raise ProviderError("Model service transport failure") from None
        time.sleep(0.25 * 2 ** attempt)
    raise ProviderError("Model service exhausted retries")


PROMPTS = {
    "plan": '拆分问题为需要证据支持的少量问点。保留主体、时间、数值条件，不猜答案。返回 {"slots":[{"id":"s1","question":"..."}]}。',
    "entities": '仅提取问题里出现的实体/专名，不猜未知答案。返回 {"entities":["原文实体"]}。',
    "extract": '从给定片段抽取有原文依据的关系。不要补常识。返回 {"relations":[{"subject":"原文主体","predicate":"关系","object":"原文客体","quote":"逐字原文","time":"可选时间","condition":"可选条件"}]}。',
    "relations": '根据问题选择有直接事实或必要桥接关系的关系，保留局部支持，排除主体时间不符。返回 {"ids":["所给关系ID"]}，按有用程度排序，不杜撰ID。',
    "select": '逐个问点选出所有互补的答案事实及必要桥接事实。局部证据也保留；不要因另一片段已经支持该问点就丢弃新增事实。检查主体、时间、限定条件，不混入无关事件。候选含短片段ID及原文句子ID。仅输出编号，不抄原文，不生成解释。返回 {"evidence":[{"slot_id":"s1","passage_id":"p1","sentence_ids":[1,3]}]}。无支持返回空数组。',
    "answer": '逐问点完整回答，仅依据提供的原文和已定位证据。列举问题要覆盖原文中的所有相关项，避免把局部事实当成完整答案。缺证项单独说明，不拒答其余项。missing_slots只是选证阶段的疑点，若原文已足够就正常回答。引用用提供的短编号 [p1]；多个引用写 [p1][p2]，不要生成其他ID。逐项检查主体、时间、数字、单位、分母和比较范围。返回 {"answer":"..."}。',
    "verify": '核验回答是否遗漏问点或原文的互补事实、错配主体时间、抄错数字、引用无依据。只使用提供的原文。列表序号和问点编号不是事实数字。区分回答漏答与原文缺证：若所需事实已在passages里，仅在issues指出应补写；只有原文确实缺少必要事实，才在gaps给出原问点slot_id与可直接检索的具体中文query，不猜答案。返回 {"ok":true或false,"issues":["具体问题"],"gaps":[{"slot_id":"s1","query":"待补证的具体问题，保留主体时间"}]}。没有缺证时gaps为空数组。',

}


class JsonModel:
    def __init__(self, services):
        self.services = services

    def identity(self):
        from dataclasses import asdict
        return {task: asdict(service) for task, service in self.services.items()}

    def call(self, task, data, trace):
        service = self.services.get(task, self.services.get("default"))
        if service is None:
            raise ProviderError(f"Missing model role: {task}")
        response = post(service, "/chat/completions", {
            "model": service.model, "temperature": 0,
            "messages": [{"role": "system", "content": PROMPTS[task]},
                         {"role": "user", "content": canonical(data)}],
            "response_format": {"type": "json_object"},
        })
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ProviderError("Incomplete or refused model response")
        result = json.loads(choice["message"]["content"])
        if not isinstance(result, dict):
            raise ProviderError("Model JSON must be an object")
        trace.add("model", task=task, model=service.model, usage=response.get("usage", {}))
        return result


class Embeddings:
    def __init__(self, service, dimension):
        self.service, self.dimension = service, dimension

    def identity(self):
        return {"provider": "compatible-http", "model": self.service.model,
                "base_url": self.service.base_url, "dimension": self.dimension}

    def embed(self, texts):
        if not texts:
            return []
        output = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            result = post(self.service, "/embeddings", {"model": self.service.model, "input": batch})
            data = sorted(result["data"], key=lambda d: d["index"])
            if [x["index"] for x in data] != list(range(len(batch))):
                raise ProviderError("Embedding response omitted or duplicated an input")
            output.extend(vector(x["embedding"], self.dimension) for x in data)
        return output


class Reranker:
    """Optional compatible /rerank service; the returned permutation is validated."""
    def __init__(self, service):
        self.service = service

    def identity(self):
        return {"model": self.service.model, "base_url": self.service.base_url}

    def rank(self, question, texts):
        response = post(self.service, "/rerank", {"model": self.service.model, "query": question,
                                                 "documents": texts, "top_n": len(texts)})
        rows = response["results"]
        indices = [r["index"] for r in rows]
        if sorted(indices) != list(range(len(texts))):
            raise ProviderError("Reranker must return each candidate once")
        return indices


class SelectiveReranker:
    """Use a remote reranker only when the candidate pool justifies its cost."""
    def __init__(self, delegate, *, min_candidates=48, max_calls_per_query=1):
        if min_candidates < 1 or max_calls_per_query < 1:
            raise ValueError("Selective reranker limits must be positive")
        self.delegate = delegate
        self.min_candidates = min_candidates
        self.max_calls_per_query = max_calls_per_query
        self._question = None
        self._calls = 0

    def identity(self):
        return {"mode": "on-demand", "delegate": self.delegate.identity(),
                "min_candidates": self.min_candidates,
                "max_calls_per_query": self.max_calls_per_query}

    def _use(self, question, texts):
        if question != self._question:
            self._question, self._calls = question, 0
        if len(texts) < self.min_candidates or self._calls >= self.max_calls_per_query:
            return False
        self._calls += 1
        return True

    def rank(self, question, texts):
        return self.delegate.rank(question, texts) if self._use(question, texts) else list(range(len(texts)))

    def rank_relations(self, question, texts):
        return (self.delegate.rank_relations(question, texts)
                if self._use(question, texts) else list(range(len(texts))))
