"""Migrated structure-aware ingestion with explicit logical-document boundaries."""
from dataclasses import asdict
from pathlib import Path
from .documents import parse_file, parse_text
from .splitter import split_blocks, SPLITTER_VERSION
from ..models import Passage, digest


def passages_from_blocks(source, blocks, *, count=len, target=384, maximum=512,
                         document_id=None, title=None):
    # Call separately for each logical article. Never infer a cross-article neighbor.
    document_id = document_id or digest({"source": source})
    chunks = list(split_blocks(blocks, count, target=target, maximum=maximum))
    passages = []
    for chunk in chunks:
        metadata = {**chunk.metadata, "location": chunk.location, "splitter": SPLITTER_VERSION}
        if title:
            metadata["title"] = title
        metadata["budget_unit"] = "caller_tokenizer" if count is not len else "characters"
        pid = digest([source, document_id, chunk.text, metadata])
        parent = digest([document_id, metadata["heading_path"]])
        passages.append(Passage(pid, source, chunk.text, "zh", document_id, metadata, parent_id=parent))
    for i, p in enumerate(passages):
        p.neighbors = [passages[j].id for j in (i - 1, i + 1) if 0 <= j < len(passages)]
    return passages


def ingest_file(path, **kwargs):
    path = Path(path)
    return passages_from_blocks(str(path), parse_file(path), **kwargs)


def ingest_text(source, text, **kwargs):
    return passages_from_blocks(source, parse_text(text), **kwargs)
