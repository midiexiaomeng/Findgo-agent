"""向量后端切换的自检与回滚测试。"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from findgo.retriever import pipeline as pipeline_module
from findgo.retriever.pipeline import Pipeline


class _FakeStore:
    def __init__(self, name: str, broken: bool = False):
        self.backend_name = name
        self.collection = "images"
        self.embedding_dim = 4
        self.broken = broken
        self.closed = False

    def add(self, ids, embeddings, metadatas):
        if self.broken:
            raise RuntimeError("写入失败")

    def query(self, vector, top_k=1):
        return [SimpleNamespace(id="__other__")]

    def delete(self, ids):
        pass

    def count(self):
        return 0

    def close(self):
        self.closed = True


def build_pipeline(old_store):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._lock = threading.RLock()
    pipeline.cfg = SimpleNamespace(vector_store=SimpleNamespace(backend=old_store.backend_name))
    pipeline.store = old_store
    pipeline._bind = lambda: None
    return pipeline


def test_switch_keeps_old_when_probe_fails(monkeypatch):
    old = _FakeStore("qdrant")
    pipeline = build_pipeline(old)

    monkeypatch.setattr(
        pipeline_module,
        "create_vector_store",
        lambda cfg, backend: _FakeStore(backend, broken=True),
    )

    with pytest.raises(RuntimeError) as e:
        pipeline.switch_backend("chroma")

    assert "保留" in str(e.value) or "仍使用" in str(e.value)
    assert pipeline.store is old
    assert pipeline.cfg.vector_store.backend == "qdrant"


def test_switch_keeps_old_when_create_fails(monkeypatch):
    old = _FakeStore("qdrant")
    pipeline = build_pipeline(old)

    def boom(cfg, backend):
        raise RuntimeError("初始化失败")

    monkeypatch.setattr(pipeline_module, "create_vector_store", boom)

    with pytest.raises(RuntimeError):
        pipeline.switch_backend("chroma")

    assert pipeline.store is old


def test_switch_succeeds_when_probe_passes(monkeypatch):
    old = _FakeStore("qdrant")
    pipeline = build_pipeline(old)
    new = _FakeStore("chroma")

    monkeypatch.setattr(pipeline_module, "create_vector_store", lambda cfg, backend: new)

    message = pipeline.switch_backend("chroma")
    assert "已切换到" in message
    assert pipeline.store is new
    assert old.closed is True
    assert pipeline.cfg.vector_store.backend == "chroma"


def test_switch_same_backend_noop():
    store = _FakeStore("qdrant")
    pipeline = build_pipeline(store)
    assert "已经是" in pipeline.switch_backend("qdrant")
