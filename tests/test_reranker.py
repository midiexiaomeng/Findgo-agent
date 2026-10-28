"""LLM 重排的单元测试：不依赖真实模型，用假客户端验证逻辑。"""

from __future__ import annotations

import pytest
from findgo.retriever.config import Config
from findgo.retriever.llm.client import LLMError
from findgo.retriever.llm.reranker import LLMReranker
from findgo.retriever.vectorstore.base import VectorHit


def make_hits(n: int = 4) -> list[VectorHit]:
    return [
        VectorHit(
            id=f"id-{i}",
            score=0.4 - i * 0.03,
            metadata={"filename": f"{i}.jpg", "caption": f"描述{i}"},
        )
        for i in range(n)
    ]


def make_cfg(**overrides) -> Config:
    cfg = Config()
    cfg.rerank.enabled = True
    cfg.rerank.backend = "ollama"
    cfg.rerank.max_candidates = 4
    cfg.rerank.batch_size = 2
    cfg.rerank.retries = 0
    for k, v in overrides.items():
        setattr(cfg.rerank, k, v)
    return cfg


def test_parse_variants():
    parse = LLMReranker._parse_scores
    assert parse('{"scores":[{"index":1,"score":85},{"index":2,"score":30}]}', 2) == {
        0: 0.85,
        1: 0.3,
    }
    assert parse('```json\n{"scores":[{"index":1,"score":90}]}\n```', 2) == {0: 0.9}
    assert parse('[{"id":1,"relevance":0.2}]', 1) == {0: 0.2}
    assert parse('{"1": 70}', 1) == {0: 0.7}
    assert parse("no json here", 2) == {}
    # 越界 index 应被忽略
    assert parse('{"scores":[{"index":9,"score":99}]}', 2) == {}


def test_rerank_uses_llm_scores():
    cfg = make_cfg()
    reranker = LLMReranker(cfg)
    hits = make_hits(4)

    def fake_chat(messages, json_mode=False, max_tokens=None):
        # 每批 2 个：把批内第 2 个打高分，用来验证排序确实变了
        return '{"scores":[{"index":1,"score":10},{"index":2,"score":95}]}'

    reranker.client.chat = fake_chat
    # 新增的可用性探测会先拦一道，测试里直接放行
    reranker.client.is_available = lambda ttl=15.0: (True, "mock")
    ordered, report = reranker.rerank("查询", hits, top_k=4)

    assert report.used is True
    assert report.llm_scored == 4
    assert ordered[0].hit.id == "id-1"  # 第一批的第二个
    assert ordered[0].llm_score == pytest.approx(0.95)
    assert ordered[0].final_score > ordered[-1].final_score


def test_rerank_degrades_when_llm_fails():
    cfg = make_cfg()
    reranker = LLMReranker(cfg)
    hits = make_hits(4)

    def boom(*args, **kwargs):
        raise LLMError("connection refused")

    reranker.client.chat = boom
    reranker.client.is_available = lambda ttl=15.0: (True, "mock")
    ordered, report = reranker.rerank("查询", hits, top_k=4)

    assert report.used is False
    assert report.errors
    # 仍然按向量分排序，不抛异常
    assert [c.hit.id for c in ordered] == [h.id for h in hits]


def test_rerank_skips_without_captions():
    cfg = make_cfg()
    reranker = LLMReranker(cfg)
    hits = [
        VectorHit(id=f"x{i}", score=0.3 - i * 0.01, metadata={"filename": f"{i}.jpg"})
        for i in range(3)
    ]

    called = {"n": 0}

    def fake_chat(*args, **kwargs):
        called["n"] += 1
        return "{}"

    reranker.client.chat = fake_chat
    _, report = reranker.rerank("查询", hits, top_k=3)

    assert called["n"] == 0
    assert report.used is False
    assert any("描述" in e for e in report.errors)


def test_rerank_respects_candidate_limit():
    cfg = make_cfg(max_candidates=4)
    reranker = LLMReranker(cfg)
    hits = make_hits(20)
    reranker.client.chat = lambda *a, **k: '{"scores":[{"index":1,"score":50}]}'
    reranker.client.is_available = lambda ttl=15.0: (True, "mock")

    _, report = reranker.rerank("查询", hits, top_k=3)
    assert report.candidates == 4
    assert len(_) == 3
