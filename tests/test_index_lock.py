"""索引文件夹级锁测试。"""

from __future__ import annotations

import threading
from pathlib import Path

from findgo.retriever.indexer import Indexer


def bare_indexer():
    indexer = object.__new__(Indexer)
    indexer._folder_locks = {}
    indexer._locks_guard = threading.Lock()
    return indexer


def test_same_folder_shares_lock(tmp_path):
    indexer = bare_indexer()
    a = indexer._folder_lock(tmp_path)
    b = indexer._folder_lock(tmp_path)
    assert a is b


def test_different_folders_get_different_locks(tmp_path):
    indexer = bare_indexer()
    a = indexer._folder_lock(tmp_path / "a")
    b = indexer._folder_lock(tmp_path / "b")
    assert a is not b


def test_index_folder_skips_when_locked(tmp_path):
    indexer = bare_indexer()
    lock = indexer._folder_lock(tmp_path)
    lock.acquire()
    try:
        stats = Indexer.index_folder(indexer, tmp_path)
        assert stats.locked is True
        assert stats.total == 0
    finally:
        lock.release()


def test_lock_released_after_failure(tmp_path, monkeypatch):
    indexer = bare_indexer()

    def boom(*args, **kwargs):
        raise RuntimeError("失败")

    monkeypatch.setattr(indexer, "_index_folder_locked", boom)

    try:
        Indexer.index_folder(indexer, tmp_path)
    except RuntimeError:
        pass

    # 锁应已释放，可以再次获取
    lock = indexer._folder_lock(tmp_path)
    assert lock.acquire(blocking=False) is True
    lock.release()
