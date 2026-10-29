"""约束优先检索测试：解析、硬过滤、多样性、理由。"""

from __future__ import annotations

import pytest
from findgo.agent.constraints import (
    Constraints,
    english_terms,
    filter_by_constraints,
    matches_term,
    parse_from_text,
)

# ── 约束解析 ──────────────────────────────────────────────────────────────


def test_parse_negative_and_preference():
    c = parse_from_text("找白色的猫，不要黑色的，最好躺着的")
    assert "黑色的" in c.must_not
    assert "躺着的" in c.should


def test_parse_empty():
    c = parse_from_text("")
    assert c.is_empty()
    assert c.core_query == ""


def test_constraints_roundtrip():
    c = Constraints(
        core_query="猫",
        must=["白色"],
        must_not=["黑色"],
        should=["躺着"],
        diversity=False,
        retrieve_count=2,
    )
    restored = Constraints.from_dict(c.to_dict())
    assert restored.core_query == "猫"
    assert restored.must == ["白色"]
    assert restored.must_not == ["黑色"]
    assert restored.should == ["躺着"]
    assert restored.diversity is False
    assert restored.retrieve_count == 2


def test_constraints_from_dict_tolerates_bad_input():
    c = Constraints.from_dict({"must": "not-a-list", "retrieve_count": "abc"})
    assert c.must == []
    assert c.retrieve_count == 0


# ── 中英匹配 ──────────────────────────────────────────────────────────────


def test_english_terms_color():
    assert "white" in english_terms("白色")
    assert "black" in english_terms("黑色")


def test_english_terms_compound():
    terms = english_terms("白色猫")
    assert "white" in terms
    assert "cat" in terms


def test_english_terms_unknown_falls_back():
    assert english_terms("量子态") == ("量子态",)


def test_matches_color():
    assert matches_term("a white cat on sofa", "白色") is True
    assert matches_term("a black cat on sofa", "白色") is False


def test_matches_action():
    assert matches_term("a cat lying on the floor", "躺着") is True
    assert matches_term("a cat standing", "躺着") is False


def test_matches_empty_caption():
    assert matches_term("", "白色") is False


# ── 硬过滤 ────────────────────────────────────────────────────────────────


def _items():
    return [
        {"path": "a.jpg", "caption": "a white cat lying on sofa"},
        {"path": "b.jpg", "caption": "a black cat sitting"},
        {"path": "c.jpg", "caption": ""},
    ]


def test_filter_must_not():
    kept, dropped = filter_by_constraints(_items(), must_not=["黑色"], must=[])
    assert [i["path"] for i in kept] == ["a.jpg", "c.jpg"]
    assert dropped[0]["path"] == "b.jpg"
    assert "黑色" in dropped[0]["_drop_reason"]


def test_filter_must_is_soft():
    """must 不做淘汰，只标记命中：BLIP 描述常不写属性。"""
    kept, dropped = filter_by_constraints(_items(), must_not=[], must=["白色"])
    assert [i["path"] for i in kept] == ["a.jpg", "b.jpg", "c.jpg"]
    assert dropped == []

    hit = next(i for i in kept if i["path"] == "a.jpg")
    assert hit["_must_hit"] == ["白色"]


def test_filter_must_not_still_hard():
    kept, dropped = filter_by_constraints(_items(), must_not=["黑色"], must=["白色"])
    assert "b.jpg" not in [i["path"] for i in kept]
    assert dropped[0]["path"] == "b.jpg"


def test_filter_keeps_uncaptioned():
    """没有描述的图不能因为「无法判断」被淘汰。"""
    kept, _ = filter_by_constraints(_items(), must_not=["黑色"], must=["白色"])
    assert "c.jpg" in [i["path"] for i in kept]


def test_filter_no_constraints():
    kept, dropped = filter_by_constraints(_items(), [], [])
    assert len(kept) == 3
    assert dropped == []


# ── 重排侧 ────────────────────────────────────────────────────────────────


def test_reranker_applies_hard_constraints():
    from findgo.retriever.llm.reranker import LLMReranker
    from findgo.retriever.vectorstore.base import VectorHit

    hits = [
        VectorHit(id="1", score=0.9, metadata={"path": "a.jpg", "caption": "white cat"}),
        VectorHit(id="2", score=0.85, metadata={"path": "b.jpg", "caption": "black cat"}),
    ]

    kept, dropped = LLMReranker._apply_hard_constraints(hits, [], ["黑色"])
    assert len(kept) == 1
    assert kept[0].id == "1"
    assert dropped[0].dropped_by == "黑色"
    assert "排除" in dropped[0].reason


def test_reranker_prompt_includes_constraints():
    from findgo.retriever.llm.reranker import LLMReranker
    from findgo.retriever.vectorstore.base import VectorHit

    reranker = LLMReranker.__new__(LLMReranker)
    hits = [VectorHit(id="1", score=0.9, metadata={"path": "a.jpg", "caption": "white cat"})]

    messages = reranker._build_prompt(
        "猫",
        hits,
        {"must": ["白色"], "must_not": ["黑色"], "should": ["躺着"]},
    )
    user = messages[-1]["content"]

    assert "必须满足" in user
    assert "白色" in user
    assert "必须排除" in user
    assert "黑色" in user
    assert "加分项" in user


def test_rerank_range_not_fixed():
    """候选少于两倍基准时应该全部参与重排。"""
    import inspect

    from findgo.retriever.llm.reranker import LLMReranker

    src = inspect.getsource(LLMReranker.rerank)
    assert "base * 2" in src


# ── 任务结构 ──────────────────────────────────────────────────────────────


def test_retrieval_args_accept_constraints():
    """工具参数 schema 必须收 constraints，否则 extra=forbid 会直接拒掉。"""
    from findgo.tools.image_retrieval import ImageRetrievalArgs

    args = ImageRetrievalArgs(
        query="猫",
        constraints={
            "core_query": "猫",
            "must": ["白色"],
            "must_not": ["黑色"],
            "should": [],
            "diversity": True,
            "retrieve_count": 2,
        },
    )
    assert args.constraints["must_not"] == ["黑色"]
    assert args.constraints["retrieve_count"] == 2


def test_retrieval_args_constraints_optional():
    from findgo.tools.image_retrieval import ImageRetrievalArgs

    args = ImageRetrievalArgs(query="猫")
    assert args.constraints == {}


def test_task_spec_carries_constraints():
    from findgo.utils.schemas import TaskSpec

    task = TaskSpec(
        query="猫",
        constraints={"core_query": "猫", "must_not": ["黑色"], "retrieve_count": 2},
    )
    assert task.constraints["must_not"] == ["黑色"]
    assert task.constraints["retrieve_count"] == 2


def test_task_spec_default_constraints_empty():
    from findgo.utils.schemas import TaskSpec

    assert TaskSpec().constraints == {}


def test_recall_limit_capped():
    """排除列表很长时，单次召回量必须被 max_recall_limit 截住。"""
    import inspect

    from findgo.retriever.searcher import Searcher

    src = inspect.getsource(Searcher._recall)
    assert "max_recall_limit" in src
    assert "limit = min(limit" in src


def test_diversity_section_exists():
    from findgo.tools.image_retrieval import ImageRetrievalTool

    assert hasattr(ImageRetrievalTool, "_diversify")
