"""Offline source/fact diagnostics; gold annotations never enter the query pipeline."""


def diagnostics(question, result, records):
    by_id = {r["passage"]["id"]: r["passage"] for r in records}
    candidate_ids = set()
    has_candidates = False
    for event in result["trace"]["events"]:
        if event["stage"] == "fusion":
            candidate_ids.update(event["ids"])
            has_candidates = True
        elif event["stage"] == "repair":
            candidate_ids.update(event["added"])
    final = result["passages"]
    candidate = [by_id[pid] for pid in candidate_ids if pid in by_id]
    gold_sources = set(question.get("sources", []))
    def coverage(passages):
        if not gold_sources:
            return None
        hits = gold_sources.intersection(p["source"] for p in passages)
        return {"hit": sorted(hits), "missing": sorted(gold_sources - hits),
                "recall": len(hits) / len(gold_sources), "all": hits == gold_sources}
    output = {"candidate_sources": coverage(candidate) if has_candidates else None,
              "final_sources": coverage(final)}
    if question.get("facts"):
        # Optional human-labelled literal facts, separate from semantic correctness.
        facts = question["facts"]
        if any(not isinstance(f.get("quote"), str) or not f["quote"].strip() for f in facts):
            raise ValueError("Fact annotations require a nonempty literal quote")
        def hits(passages):
            return [fact["id"] for fact in facts if any(
                p["source"] == fact["source"] and fact["quote"] in p["text"] for p in passages)]
        output["literal_facts"] = {"expected": [f["id"] for f in facts],
                                   "candidate_hit": hits(candidate) if has_candidates else None,
                                   "final_hit": hits(final)}
    return output
