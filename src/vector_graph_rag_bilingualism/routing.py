"""Explicit language routing; mixed collections never silently lose one language."""


class Router:
    def __init__(self, pipelines):
        if not pipelines or set(pipelines) - {"zh", "en"}:
            raise ValueError("Register zh/en pipelines")
        self.pipelines = pipelines

    def query(self, question, *, language="auto", **kwargs):
        if language == "both":
            raise NotImplementedError("Joint cross-language retrieval is a separate future profile")
        if language == "auto":
            if len(self.pipelines) != 1:
                raise ValueError("Mixed knowledge base: explicitly choose zh or en")
            language = next(iter(self.pipelines))
        if language not in self.pipelines:
            raise ValueError("Requested language is not indexed")
        return self.pipelines[language].query(question, **kwargs)
