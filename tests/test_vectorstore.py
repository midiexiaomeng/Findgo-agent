"""向量库抽象层的契约测试：两个后端必须表现完全一致。"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from findgo.retriever.config import Config
from findgo.retriever.vectorstore import create_vector_store, l2_normalize


def make_store(tmp_path, backend: str, dim: int = 8):
    cfg = Config()
    cfg.vector_store.backend = backend
    cfg.vector_store.collection = "images"
    cfg.embedding.embedding_dim = dim
    if backend == "chromadb":
        cfg.vector_store.chromadb.path = str(tmp_path / "chroma")
    else:
        cfg.vector_store.qdrant.mode = "local"
        cfg.vector_store.qdrant.path = str(tmp_path / "qdrant")
    return create_vector_store(cfg)


@pytest.fixture(params=["chromadb", "qdrant"])
def store(request, tmp_path):
    s = make_store(tmp_path, request.param)
    yield s
    s.close()


def test_add_query_count(store):
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"img-{i}")) for i in range(5)]
    vectors = l2_normalize(np.random.RandomState(0).rand(5, 8).astype("float32"))
    metas = [{"path": f"D:/p/{i}.jpg", "filename": f"{i}.jpg", "caption": f"图片{i}"} for i in range(5)]

    store.add(ids, vectors, metas)
    assert store.count() == 5
    assert set(store.get_all_ids()) == set(ids)

    hits = store.query(vectors[2], top_k=3)
    assert len(hits) == 3
    assert hits[0].id == ids[2]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].metadata["filename"] == "2.jpg"


def test_upsert_is_idempotent(store):
    vid = str(uuid.uuid5(uuid.NAMESPACE_URL, "one"))
    vec = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
    store.add([vid], vec, [{"path": "old", "filename": "old.jpg"}])
    store.add([vid], vec, [{"path": "new", "filename": "new.jpg"}])

    assert store.count() == 1
    got = store.get([vid])[0]
    assert got.metadata["path"] == "new"


def test_delete_and_reset(store):
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"d-{i}")) for i in range(3)]
    vec = l2_normalize(np.random.rand(3, 8).astype("float32"))
    store.add(ids, vec, [{"path": str(i)} for i in range(3)])

    store.delete([ids[0]])
    assert store.count() == 2

    store.reset()
    assert store.count() == 0
    assert store.query(vec[0], top_k=5) == []


def test_dimension_mismatch_raises(store):
    from findgo.retriever.vectorstore import VectorStoreError

    with pytest.raises(VectorStoreError):
        store.add(["x"], np.ones((1, 4), dtype=np.float32), [{"path": "x"}])

    with pytest.raises(VectorStoreError):
        store.query(np.ones(4, dtype=np.float32), top_k=1)


def test_where_filter(store):
    ids = ["aaaa1111-0000-0000-0000-00000000000" + str(i) for i in range(3)]
    vec = l2_normalize(np.random.rand(3, 8).astype("float32"))
    metas = [
        {"path": "a", "filename": "a.jpg", "ext": ".jpg"},
        {"path": "b", "filename": "b.png", "ext": ".png"},
        {"path": "c", "filename": "c.jpg", "ext": ".jpg"},
    ]
    store.add(ids, vec, metas)
    hits = store.query(vec[0], top_k=5, where={"ext": ".jpg"})
    assert {h.metadata["filename"] for h in hits} == {"a.jpg", "c.jpg"}
