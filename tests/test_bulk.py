"""大结果集与“全部”语义测试。"""

from __future__ import annotations

import pytest
from findgo.retriever.llm.reranker import LLMReranker, ScoredCandidate
from findgo.retriever.vectorstore.base import VectorHit
from findgo.tools import ImageRetrievalTool
from findgo.utils.schemas import TaskSpec


def _cand(score: float, idx: int) -> ScoredCandidate:
    hit = VectorHit(id=str(idx), score=score, metadata={"path": f"D:/a/{idx}.jpg"})
    return ScoredCandidate(
        hit=hit, vec_score=score, vec_norm=score, llm_score=None, final_score=score
    )


def test_finish_extends_with_tail():
    head = [_cand(0.9, i) for i in range(3)]
    tail = [_cand(0.5, 100 + i) for i in range(10)]
    out = LLMReranker._finish(head, 8, tail)
    assert len(out) == 8
    assert out[0].final_score == pytest.approx(0.9)


def test_finish_without_tail():
    head = [_cand(0.9, i) for i in range(3)]
    out = LLMReranker._finish(head, 8)
    assert len(out) == 3


def test_finish_respects_top_k():
    head = [_cand(0.9 - i * 0.01, i) for i in range(20)]
    tail = [_cand(0.5, 100 + i) for i in range(10)]
    out = LLMReranker._finish(head, 5, tail)
    assert len(out) == 5


class _Resp:
    results = []

    @staticmethod
    def status_line() -> str:
        return "0"


class _RecordingSearcher:
    def __init__(self):
        self.last: dict = {}

    def search_text(
        self,
        query,
        top_k=8,
        scope=None,
        all_results=False,
        should_stop=None,
        exclude_paths=None,
        constraints=None,
    ):
        self.last = {
            "query": query,
            "top_k": top_k,
            "scope": scope,
            "all_results": all_results,
        }
        return _Resp()


class _Pipeline:
    def __init__(self):
        self.searcher = _RecordingSearcher()


def test_tool_passes_all_results():
    pipeline = _Pipeline()
    tool = ImageRetrievalTool(pipeline)
    tool.run("所有猫的图片", all_results=True)
    assert pipeline.searcher.last["all_results"] is True


def test_tool_accepts_large_top_k():
    pipeline = _Pipeline()
    tool = ImageRetrievalTool(pipeline)
    payload = tool.validate({"query": "猫", "top_k": 1000})
    assert payload["top_k"] == 1000


def test_task_spec_has_all_results():
    task = TaskSpec(id=1, query="所有猫", all_results=True)
    assert task.all_results is True
