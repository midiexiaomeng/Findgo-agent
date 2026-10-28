"""下载缓存与联网搜索工具测试。"""

from __future__ import annotations

from imgsearch_agent.downloads import DownloadCache


def test_save_and_stats(tmp_path):
    cache = DownloadCache(tmp_path, max_gb=1.0)
    cache.save("http://example.com/a.jpg", b"x" * 200)
    stats = cache.stats()
    assert stats["count"] == 1
    assert stats["bytes"] == 200


def test_enforce_limit_removes_oldest(tmp_path):
    cache = DownloadCache(tmp_path, max_gb=0.000001)
    for i in range(3):
        cache.save(f"http://example.com/{i}.jpg", b"y" * 1000)
    removed = cache.enforce_limit()
    assert removed > 0
    assert cache.stats()["bytes"] <= cache.max_bytes


def test_clear(tmp_path):
    cache = DownloadCache(tmp_path, max_gb=1.0)
    cache.save("http://example.com/a.jpg", b"z")
    assert cache.clear() == 1
    assert cache.stats()["count"] == 0


def test_same_url_reuses_file(tmp_path):
    cache = DownloadCache(tmp_path, max_gb=1.0)
    p1 = cache.save("http://example.com/a.jpg", b"1")
    p2 = cache.save("http://example.com/a.jpg", b"2")
    assert p1 == p2
    assert p1.read_bytes() == b"2"
