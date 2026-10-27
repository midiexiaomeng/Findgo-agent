"""端到端测试：用假编码器跑通 索引 -> 检索 -> 切库 全链路。

真实模型（CLIP/BLIP）不参与，验证的是工程结构：
- 索引写入的字段是否完整
- 增量索引是否正确跳过
- 以文搜图 / 以图搜图 是否能召回正确图片
- 切换向量库后同一套上层代码是否照常工作
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from imgsearch.captioner import Captioner
from imgsearch.clip_encoder import ClipEncoder
from imgsearch.config import Config
from imgsearch.indexer import Indexer
from imgsearch.llm.reranker import LLMReranker
from imgsearch.searcher import Searcher
from imgsearch.vectorstore import create_vector_store

COLORS = {
    "red.png": (255, 0, 0),
    "green.png": (0, 255, 0),
    "blue.png": (0, 0, 255),
}


class FakeEncoder:
    """把图片平均色当作 8 维向量，保证"红色"能召回红图。"""

    dim = 8
    device = "cpu"
    loaded = True

    @staticmethod
    def _vec_rgb(rgb) -> np.ndarray:
        r, g, b = [float(x) for x in rgb]
        v = np.array([r, g, b, r - g, g - b, b - r, (r + g + b) / 3.0, 1.0], dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def _vec_path(self, path) -> np.ndarray:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((4, 4))
            pixels = list(im.getdata())
        r = sum(p[0] for p in pixels) / len(pixels)
        g = sum(p[1] for p in pixels) / len(pixels)
        b = sum(p[2] for p in pixels) / len(pixels)
        return self._vec_rgb((r, g, b))

    def encode_images(self, paths):
        return np.stack([self._vec_path(p) for p in paths])

    def encode_image(self, path):
        return self._vec_path(path)

    def encode_text(self, text):
        mapping = {"红色": (255, 0, 0), "绿色": (0, 255, 0), "蓝色": (0, 0, 255)}
        return self._vec_rgb(mapping.get(text, (255, 0, 0)))

    def unload(self):
        pass


class FakeCaptioner:
    available = True
    enabled = True

    def captions_for(self, records):
        return {r["id"]: f"一张 {Path(r['path']).stem} 颜色的图片" for r in records}

    def caption_image(self, path):
        return f"一张 {Path(path).stem} 颜色的图片"

    def unload(self):
        pass

    def cache_size(self):
        return 0

    def stats(self):
        return {"enabled": True}

    def close(self):
        pass


@pytest.fixture()
def workspace(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name, rgb in COLORS.items():
        Image.new("RGB", (64, 64), rgb).save(images / name)
    return images


def build(tmp_path, backend: str = "chromadb"):
    cfg = Config()
    cfg.vector_store.backend = backend
    cfg.vector_store.collection = "images"
    cfg.embedding.embedding_dim = 8
    cfg.caption.cache_db = str(tmp_path / "captions.sqlite")
    cfg.search.rerank_candidates = 4
    cfg.search.top_k = 3
    if backend == "chromadb":
        cfg.vector_store.chromadb.path = str(tmp_path / "chroma")
    else:
        cfg.vector_store.qdrant.mode = "local"
        cfg.vector_store.qdrant.path = str(tmp_path / "qdrant")

    store = create_vector_store(cfg)
    encoder = FakeEncoder()
    captioner = FakeCaptioner()
    indexer = Indexer(cfg, encoder, store, captioner)
    searcher = Searcher(cfg, encoder, store, LLMReranker(cfg), captioner)
    return cfg, store, indexer, searcher


@pytest.mark.parametrize("backend", ["chromadb", "qdrant"])
def test_index_and_search(workspace, tmp_path, backend):
    cfg, store, indexer, searcher = build(tmp_path, backend)

    stats = indexer.index_folder(workspace)
    assert stats.indexed == 3
    assert store.count() == 3

    # 再次索引：全部跳过
    again = indexer.index_folder(workspace)
    assert again.indexed == 0
    assert again.skipped == 3
    assert store.count() == 3

    # 以文搜图（关闭 LLM，纯向量）
    resp = searcher.search_text("红色", top_k=3, use_rerank=False)
    assert resp.results
    assert resp.results[0].filename == "red.png"
    assert resp.results[0].exists is True
    assert resp.results[0].caption  # 描述已落库

    # 以图搜图：排除自身
    resp2 = searcher.search_image(workspace / "red.png", top_k=3, use_rerank=False, exclude_self=True)
    names = [r.filename for r in resp2.results]
    assert "red.png" not in names
    assert names

    store.close()


def test_switch_backend_keeps_working(workspace, tmp_path):
    cfg, store, indexer, searcher = build(tmp_path, "chromadb")
    indexer.index_folder(workspace)
    assert store.count() == 3

    # 换到 qdrant：数据不共享，需要重新索引
    cfg.vector_store.qdrant.mode = "local"
    cfg.vector_store.qdrant.path = str(tmp_path / "qdrant2")
    store2 = create_vector_store(cfg, backend="qdrant")
    indexer2 = Indexer(cfg, FakeEncoder(), store2, FakeCaptioner())
    searcher2 = Searcher(cfg, FakeEncoder(), store2, LLMReranker(cfg), FakeCaptioner())

    assert store2.count() == 0
    indexer2.index_folder(workspace)
    assert store2.count() == 3
    resp = searcher2.search_text("蓝色", top_k=3, use_rerank=False)
    assert resp.results[0].filename == "blue.png"

    store.close()
    store2.close()


def test_missing_file_is_flagged(workspace, tmp_path):
    cfg, store, indexer, searcher = build(tmp_path)
    indexer.index_folder(workspace)
    (workspace / "red.png").unlink()

    resp = searcher.search_text("红色", top_k=3, use_rerank=False)
    assert resp.results[0].filename == "red.png"
    assert resp.results[0].exists is False
    store.close()
