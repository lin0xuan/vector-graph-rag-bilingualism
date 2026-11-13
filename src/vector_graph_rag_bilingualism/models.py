"""Stable evidence contracts, deliberately independent of model/storage SDKs."""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from uuid import uuid4


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


@dataclass
class Passage:
    id: str
    source: str
    text: str
    language: str
    document_id: str
    metadata: dict = field(default_factory=dict)
    neighbors: list[str] = field(default_factory=list)
    parent_id: str = ""


@dataclass
class QuestionSlot:
    id: str
    question: str


@dataclass
class Evidence:
    passage_id: str
    slot_id: str
    quote: str
    start: int
    end: int


@dataclass
class Trace:
    id: str = field(default_factory=lambda: uuid4().hex)
    events: list[dict] = field(default_factory=list)

    def add(self, stage, **data):
        self.events.append({"stage": stage, **data})


@dataclass
class Result:
    question: str
    language: str
    profile: str
    snapshot_id: str
    status: str
    answer: str = ""
    passages: list[Passage] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)

    def as_dict(self):
        return asdict(self)


def vector(value, dimension=None):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Embedding must be a nonempty vector")
    result = [float(x) for x in value]
    if any(not math.isfinite(x) for x in result) or not any(result):
        raise ValueError("Embedding must be finite and nonzero")
    if dimension is not None and len(result) != dimension:
        raise ValueError("Embedding dimension mismatch")
    return result
