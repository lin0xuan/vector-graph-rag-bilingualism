"""Explicit profiles; manifests never contain credentials."""
from dataclasses import asdict, dataclass, field
from .models import digest

UPSTREAM_COMMIT = "47590ad4eab5838cdcfec94f8a53ebfe70ebe330"
PIPELINE_VERSION = "coverage-v2"


@dataclass(frozen=True)
class ModelService:
    base_url: str
    model: str
    key_env: str
    timeout: float = 60

    def __post_init__(self):
        from urllib.parse import urlsplit
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.query or url.fragment:
            raise ValueError("Use an HTTP(S) endpoint without embedded credentials/query")
        if not self.model or not self.key_env.isidentifier() or self.timeout <= 0:
            raise ValueError("Invalid model service")


@dataclass(frozen=True)
class ChineseConfig:
    retrieval_mode: str = "two-stage"
    candidate_limit: int = 100
    per_channel: int = 30
    max_slots: int = 6
    max_repairs: int = 2
    context_budget: int = 8000
    selection_batch: int = 16
    graph_hops: int = 1
    profile: str = "zh_coverage_v2"

    def __post_init__(self):
        if self.retrieval_mode not in ("single-stage", "two-stage"):
            raise ValueError("Unknown Chinese retrieval mode")
        if not 0 <= self.max_repairs <= 2 or not 1 <= self.graph_hops <= 2:
            raise ValueError("Invalid repair or graph budget")
        for key in ("candidate_limit", "per_channel", "max_slots", "context_budget", "selection_batch"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")

    def fingerprint(self):
        return digest({"version": PIPELINE_VERSION, **asdict(self)})


@dataclass(frozen=True)
class EnglishConfig:
    settings: dict = field(default_factory=dict)
    profile: str = "en_upstream_jev_1_13"

    def __post_init__(self):
        forbidden = {k for k in self.settings if any(s in k.lower() for s in ("key", "token", "password"))}
        if forbidden:
            raise ValueError("Keep credentials in environment, not profile settings")
        if self.settings.get("reranker_model", "jev-1.13.0") != "jev-1.13.0":
            raise ValueError("This English profile pins jev-1.13.0")
        if any(isinstance(v, dict) for v in self.settings.values()):
            raise ValueError("Nested English service settings require a separate validated profile")
