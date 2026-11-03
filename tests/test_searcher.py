"""Searcher 的单元测试：只验证调度与过滤逻辑，不加载任何真实模型。

``tests/test_index_search.py`` 已经覆盖「真实向量库 + 假编码器」的端到端链路，
这里补齐它没碰到的边界：空查询、目录限定、上一轮排除、暂存过滤、相似度阈值、
全部结果，以及以图搜图在描述缺失时的降级路径。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from findgo.retriever.config import Config
from findgo.retriever.llm.reranker import RerankReport, ScoredCandidate
from findgo.retriever.searcher import Searcher
from findgo.retriever.vectorstore.base import VectorHit

DIM = 4


class StubEncoder:
    """只记录调用；向量本身不参与断言。"""

    device = "cpu"
    loaded = True

    def __init__(self) -> None:
        self.text_calls: list[str] = []
        self.image_calls: list[str] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.text_calls.append(text)
        return np.ones(DIM, dtype=np.float32)

    def encode_image(self, path) -> np.ndarray:
        self.image_calls.append(str(path))
        return np.ones(DIM, dtype=np.float32)


class StubStore:
    """按写入顺序返回命中，并记录 query 参数。"""

    backend_name = "stub"
    collection = "images"
    embedding_dim = DIM

    def __init__(self, hits: list[VectorHit]) -> None:
        self.hits = list(hits)
        self.query_calls: list[dict] = []

    def query(self, embedding, top_k: int, where=None) -> list[VectorHit]:
        self.query_calls.append({"top_k": top_k, "where": where})
        return self.hits[:top_k]

    def count(self) -> int:
        return len(self.hits)

    def stats(self) -> dict:
        return {
            "backend": self.backend_name,
            "collection": self.collection,
            "count": len(self.hits),
            "embedding_dim": self.embedding_dim,
        }


class StubReranker:
    """只截断与记录，真实排序由 tests/test_reranker.py 负责。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.client = SimpleNamespace(
            backend="stub",
            model="stub-model",
            describe_backend=lambda: "stub",
        )

    def rerank(
        self,
        query: str,
        hits,
        top_k: int | None = None,
        use_llm: bool = True,
        should_stop=None,
        constraints: dict | None = None,
    ):
        hits = list(hits)
        self.calls.append(
            {
                "query": query,
                "hits": hits,
                "top_k": top_k,
                "use_llm": use_llm,
                "should_stop": should_stop,
                "constraints": constraints,
            }
        )
        take = len(hits) if top_k is None else int(top_k)
        scored = [
            ScoredCandidate(
                hit=hit,
                vec_score=float(hit.score),
                vec_norm=0.0,
                llm_score=None,
                final_score=float(hit.score),
            )
            for hit in hits[:take]
        ]
        return scored, RerankReport(used=bool(use_llm), candidates=len(hits))


class StubCaptioner:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.text = "一只猫趴在窗台上"
        self.calls: list[str] = []

    def caption_image(self, path) -> str:
        self.calls.append(str(path))
        if self.error is not None:
            raise self.error
        return self.text

    def stats(self) -> dict:
        return {"enabled": True}


def make_hit(idx: int, path, score: float, staged: bool = False) -> VectorHit:
    meta = {
        "path": str(path),
        "filename": str(path).replace("\\", "/").rsplit("/", 1)[-1],
        "caption": f"描述{idx}",
    }
    if staged:
        meta["is_staged"] = True
    return VectorHit(id=f"id{idx}", score=score, metadata=meta)


def build(hits, *, caption_enabled: bool = True, top_k: int = 3, recall_k: int = 50):
    cfg = Config()
    cfg.search.top_k = top_k
    cfg.search.recall_k = recall_k
    cfg.caption.enabled = caption_enabled
    encoder = StubEncoder()
    store = StubStore(hits)
    reranker = StubReranker()
    captioner = StubCaptioner()
    searcher = Searcher(cfg, encoder, store, reranker, captioner)
    return searcher, encoder, store, reranker, captioner


# ── 以文搜图 ──────────────────────────────────────────────────────────────────


def test_empty_query_returns_early_without_touching_backend():
    searcher, encoder, store, reranker, _ = build([])

    resp = searcher.search_text("   ")

    assert resp.results == []
    assert resp.note == "查询为空"
    assert encoder.text_calls == []
    assert store.query_calls == []
    assert reranker.calls == []


def test_empty_query_with_constraints_still_searches(tmp_path):
    """结构化筛选可以在没有关键词时单独成立，不能按空查询短路。"""
    searcher, encoder, _, reranker, _ = build([make_hit(0, tmp_path / "a.jpg", 0.8)])
    constraints = {"must": ["猫"]}

    resp = searcher.search_text("", constraints=constraints)

    assert encoder.text_calls == [""]
    assert reranker.calls[0]["constraints"] == constraints
    assert len(resp.results) == 1


def test_text_search_maps_hits_to_ranked_results(tmp_path):
    present = tmp_path / "cat.jpg"
    present.write_bytes(b"x")
    searcher, _, _, reranker, _ = build(
        [make_hit(0, present, 0.9), make_hit(1, tmp_path / "missing.jpg", 0.5)]
    )

    resp = searcher.search_text("猫")

    assert [r.rank for r in resp.results] == [1, 2]
    assert resp.results[0].filename == "cat.jpg"
    assert resp.results[0].caption == "描述0"
    assert resp.results[0].exists is True
    assert resp.results[1].exists is False
    assert resp.mode == "text"
    assert resp.recall_count == 2
    assert reranker.calls[0]["top_k"] == 3
    assert "召回 2 条" in resp.status_line()
    assert "LLM 精排 0/2" in resp.status_line()


def test_min_score_drops_weak_hits(tmp_path):
    searcher, *_ = build(
        [make_hit(0, tmp_path / "a.jpg", 0.9), make_hit(1, tmp_path / "b.jpg", 0.2)]
    )

    resp = searcher.search_text("猫", min_score=0.5)

    assert [r.filename for r in resp.results] == ["a.jpg"]


def test_scope_limits_to_path_prefix_and_over_recalls(tmp_path):
    inside = tmp_path / "cats" / "a.jpg"
    outside = tmp_path / "dogs" / "b.jpg"
    inside.parent.mkdir()
    outside.parent.mkdir()
    searcher, _, store, *_ = build([make_hit(0, inside, 0.9), make_hit(1, outside, 0.8)])

    resp = searcher.search_text("猫", scope=str(tmp_path / "cats"))

    assert [r.filename for r in resp.results] == ["a.jpg"]
    # 限定目录时先多召回再过滤，否则过滤后可能凑不满 top_k
    assert store.query_calls[0]["top_k"] > 2


def test_exclude_paths_skips_previous_round_case_insensitively(tmp_path):
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    searcher, *_ = build([make_hit(0, first, 0.9), make_hit(1, second, 0.8)])

    resp = searcher.search_text("猫", exclude_paths={str(first).upper()})

    assert [r.filename for r in resp.results] == ["b.jpg"]


def test_staged_hits_are_filtered_out(tmp_path):
    searcher, _, store, *_ = build(
        [make_hit(0, tmp_path / "a.jpg", 0.9), make_hit(1, tmp_path / "b.jpg", 0.8, staged=True)]
    )

    resp = searcher.search_text("猫")

    assert [r.filename for r in resp.results] == ["a.jpg"]
    # 软删除条件同时下推给后端，减少无效召回
    assert store.query_calls[0]["where"] == {"is_staged": {"$ne": True}}


def test_all_results_widens_top_k_to_store_size(tmp_path):
    hits = [make_hit(i, tmp_path / f"{i}.jpg", 0.9 - i * 0.01) for i in range(6)]
    searcher, _, _, reranker, _ = build(hits, top_k=2, recall_k=2)

    resp = searcher.search_text("猫", all_results=True)

    assert reranker.calls[0]["top_k"] == 6
    assert len(resp.results) == 6


# ── 以图搜图 ──────────────────────────────────────────────────────────────────


def test_image_search_excludes_self_and_requests_one_extra_hit(tmp_path):
    me = tmp_path / "me.jpg"
    me.write_bytes(b"x")
    searcher, encoder, store, reranker, _ = build(
        [make_hit(0, me, 0.99), make_hit(1, tmp_path / "other.jpg", 0.8)],
        recall_k=2,
    )

    resp = searcher.search_image(me, top_k=2, use_rerank=False)

    assert encoder.image_calls == [str(me)]
    assert [r.filename for r in resp.results] == ["other.jpg"]
    assert resp.recall_count == 1
    assert store.query_calls[0]["top_k"] >= 3
    assert reranker.calls[0]["use_llm"] is False


def test_image_search_uses_caption_as_semantic_query(tmp_path):
    me = tmp_path / "me.jpg"
    me.write_bytes(b"x")
    searcher, _, _, reranker, captioner = build([make_hit(1, tmp_path / "other.jpg", 0.8)])
    captioner.text = "一只橘猫趴在窗台"

    resp = searcher.search_image(me)

    assert captioner.calls == [str(me)]
    assert resp.query_caption == "一只橘猫趴在窗台"
    assert reranker.calls[0]["query"] == "一只橘猫趴在窗台"
    assert reranker.calls[0]["use_llm"] is True


def test_image_search_degrades_when_caption_fails(tmp_path):
    me = tmp_path / "me.jpg"
    me.write_bytes(b"x")
    searcher, _, _, reranker, captioner = build([make_hit(1, tmp_path / "other.jpg", 0.8)])
    captioner.error = RuntimeError("显存不足")

    resp = searcher.search_image(me)

    assert "描述生成失败" in resp.note
    assert "跳过 LLM 重排" in resp.note
    assert reranker.calls[0]["use_llm"] is False
    assert reranker.calls[0]["query"] == ""
    # 降级后仍要给出向量结果，不能返回空列表
    assert [r.filename for r in resp.results] == ["other.jpg"]


def test_image_search_without_caption_config_skips_llm(tmp_path):
    me = tmp_path / "me.jpg"
    me.write_bytes(b"x")
    searcher, _, _, reranker, captioner = build(
        [make_hit(1, tmp_path / "other.jpg", 0.8)], caption_enabled=False
    )

    resp = searcher.search_image(me)

    assert captioner.calls == []
    assert "未启用 BLIP" in resp.note
    assert reranker.calls[0]["use_llm"] is False


# ── 统计 ──────────────────────────────────────────────────────────────────────


def test_stats_merges_store_and_runtime_info(tmp_path):
    searcher, *_ = build([make_hit(0, tmp_path / "a.jpg", 0.9)])

    info = searcher.stats()

    assert info["images"] == 1
    assert info["backend"] == "stub"
    assert info["collection"] == "images"
    assert info["dim"] == DIM
    assert info["device"] == "cpu"
    assert info["encoder_loaded"] is True
    assert info["rerank_enabled"] is True
    assert info["caption"] == {"enabled": True}


def test_stats_survives_store_without_detailed_stats(tmp_path):
    """后端只实现最小契约时，统计页不应整页 KeyError。"""

    class MinimalStore(StubStore):
        def stats(self) -> dict:
            return {}

    cfg = Config()
    store = MinimalStore([make_hit(0, tmp_path / "a.jpg", 0.9)])
    searcher = Searcher(cfg, StubEncoder(), store, StubReranker(), StubCaptioner())

    info = searcher.stats()

    assert info["images"] == 1
    assert info["backend"] == "stub"
    assert info["collection"] == "images"
    assert info["dim"] == DIM
