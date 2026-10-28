"""描述缓存容量上限测试。"""

from __future__ import annotations

from findgo.retriever.captioner import Captioner
from findgo.retriever.config import Config


def make_captioner(tmp_path, limit=5):
    cfg = Config.from_yaml()
    cfg.caption.cache_db = str(tmp_path / "captions.sqlite")
    cfg.caption.max_cache_rows = limit
    return Captioner(cfg)


def test_cache_evicts_when_over_limit(tmp_path):
    captioner = make_captioner(tmp_path, limit=5)
    captioner._cache_put_many([(f"id{i}", "描述", 1.0) for i in range(10)])
    assert captioner.cache_size() == 5


def test_cache_keeps_recent_when_under_limit(tmp_path):
    captioner = make_captioner(tmp_path, limit=20)
    captioner._cache_put_many([(f"id{i}", "描述", 1.0) for i in range(8)])
    assert captioner.cache_size() == 8


def test_zero_limit_disables_eviction(tmp_path):
    captioner = make_captioner(tmp_path, limit=0)
    captioner._cache_put_many([(f"id{i}", "描述", 1.0) for i in range(6)])
    assert captioner.cache_size() == 6
