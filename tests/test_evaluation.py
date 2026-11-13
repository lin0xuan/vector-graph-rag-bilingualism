import pytest
from vector_graph_rag_bilingualism import ChinesePipeline, SnapshotStore
from vector_graph_rag_bilingualism.evaluation import EvaluationRunner, read_rows
from helpers import FakeEmbedding, FakeModel, corpus


def test_scoring_resume_reuses_answer_and_targeted_retry_only_one(tmp_path):
    model = FakeModel()
    pipeline = ChinesePipeline(SnapshotStore(tmp_path / "index"), "zh", FakeEmbedding(), model)
    snapshot = pipeline.build(corpus())
    calls = []
    fail = [True]
    def judge(q, result):
        calls.append(q["id"])
        if fail[0]:
            raise TimeoutError()
        return "fixture-pass"
    runner = EvaluationRunner(tmp_path / "eval", pipeline, snapshot, judge=judge, judge_identity="fixture-v1")
    questions = [{"id": "one", "question": "谁审批，期限多久？"}, {"id": "two", "question": "谁审批，期限几天？"}]
    first = runner.run(questions)
    assert first["graded"] == 0
    generated = model.calls["answer"]
    fail[0] = False
    second = runner.run(questions, resume=True)
    assert second["graded"] == 2
    assert model.calls["answer"] == generated
    third = runner.run(questions, resume=True, retry_ids=["two"])
    assert third["questions"] == 2 and third["graded"] == 2
    assert model.calls["answer"] == generated + 1
    rows = read_rows(tmp_path / "eval" / "generations.jsonl")
    assert [r["id"] for r in rows] == ["one", "two", "two"]
    before = model.calls.copy()
    with pytest.raises(ValueError, match="manifest"):
        runner.run(questions[:1], resume=True)
    assert model.calls == before


def test_torn_last_record_is_recoverable_but_committed_corruption_is_not(tmp_path):
    path = tmp_path / "generations.jsonl"
    path.write_bytes(b'{"id":"done"}\n{"id":"interrup')
    assert read_rows(path) == [{"id": "done"}]
    assert path.read_bytes() == b'{"id":"done"}\n'
    path.write_bytes(b'{"id":"done"}\nBAD\n')
    with pytest.raises(ValueError, match="Corrupt"):
        read_rows(path)


def test_manifest_version_change_rejected_before_calls(tmp_path):
    model = FakeModel()
    pipeline = ChinesePipeline(SnapshotStore(tmp_path / "index"), "zh", FakeEmbedding(), model)
    snapshot = pipeline.build(corpus())
    runner = EvaluationRunner(tmp_path / "eval", pipeline, snapshot)
    questions = [{"id": "one", "question": "谁审批，期限多久？"}]
    runner.run(questions)
    before = model.calls.copy()
    model.identity = lambda: {"model": "changed"}
    with pytest.raises(ValueError, match="manifest"):
        runner.run(questions, resume=True)
    assert before == model.calls


def test_source_and_literal_fact_metrics_do_not_enter_model_prompt(tmp_path):
    model = FakeModel()
    pipeline = ChinesePipeline(SnapshotStore(tmp_path / "index"), "zh", FakeEmbedding(), model)
    snapshot = pipeline.build(corpus())
    runner = EvaluationRunner(tmp_path / "eval", pipeline, snapshot)
    questions = [{"id": "one", "question": "谁审批，期限多久？", "sources": ["policy", "time"],
                  "facts": [{"id": "duration", "source": "time", "quote": "三个工作日"}]}]
    original = model.call
    def guarded(task, data, trace):
        assert "reference" not in data and "facts" not in data and "sources" not in data
        return original(task, data, trace)
    model.call = guarded
    runner.run(questions)
    row = read_rows(tmp_path / "eval" / "generations.jsonl")[0]
    assert row["diagnostics"]["final_sources"]["all"]
    assert row["diagnostics"]["literal_facts"]["final_hit"] == ["duration"]
