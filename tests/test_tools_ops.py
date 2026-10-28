"""导出与删除索引工具测试。"""

from __future__ import annotations

from pathlib import Path

from findgo.tools import ExportResultsTool, RemoveIndexTool
from findgo.utils.sandbox import Sandbox


class _Item:
    def __init__(self, rank: int, path: str):
        self.rank = rank
        self.path = path
        self.score = 0.9
        self.caption = "一只猫"


class _Response:
    def __init__(self):
        self.results = [
            _Item(1, r"D:\a\1.jpg"),
            _Item(2, r"D:\a\2.jpg"),
        ]


class _Searcher:
    def search_text(self, query, **kwargs):
        return _Response()


class _ExportPipeline:
    searcher = _Searcher()


def test_export_csv(tmp_path, monkeypatch):
    target = tmp_path / "exports"
    monkeypatch.setattr("findgo.retriever.runtime.PROJECT_ROOT", str(tmp_path))

    tool = ExportResultsTool(
        _ExportPipeline(),
        Sandbox(image_roots=[str(tmp_path)], workspace_roots=[str(tmp_path)]),
    )
    result = tool.run("猫", top_k=5)

    saved = Path(result["path"])
    assert saved.exists()
    text = saved.read_text(encoding="utf-8-sig")
    assert "rank,path,score,caption" in text
    assert result["count"] == 2


class _Hit:
    def __init__(self, idx: str, path: str):
        self.id = idx
        self.path = path


class _Store:
    def __init__(self, folder):
        self.deleted: list[str] = []
        self.rows = [
            _Hit("1", str(folder / "a.jpg")),
            _Hit("2", str(folder / "b.jpg")),
            _Hit("3", str(folder.parent / "c.jpg")),
        ]

    def get(self, limit=None):
        return list(self.rows)

    def delete(self, ids):
        self.deleted.extend(ids)
        keep = [r for r in self.rows if r.id not in set(ids)]
        self.rows = keep

    def count(self):
        return len(self.rows)


class _RemovePipeline:
    def __init__(self, folder):
        self.store = _Store(folder)


def _guard(tmp_path):
    # 用临时目录模拟 D:\photos\x
    folder = tmp_path / "photos" / "x"
    folder.mkdir(parents=True)
    return Sandbox(image_roots=[str(tmp_path)], workspace_roots=[str(tmp_path)]), folder


def test_remove_index_dry_run(tmp_path):
    guard, folder = _guard(tmp_path)
    pipeline = _RemovePipeline(folder)
    tool = RemoveIndexTool(pipeline, guard)

    result = tool.run(str(folder))
    assert result["count"] == 2
    assert result["deleted"] is False
    assert pipeline.store.deleted == []


def test_remove_index_real_delete(tmp_path):
    guard, folder = _guard(tmp_path)
    pipeline = _RemovePipeline(folder)
    tool = RemoveIndexTool(pipeline, guard)

    result = tool.run(str(folder), dry_run=False)
    assert result["deleted"] is True
    assert pipeline.store.deleted == ["1", "2"]
    assert result["remaining"] == 1


def test_remove_index_empty(tmp_path):
    guard, folder = _guard(tmp_path)
    pipeline = _RemovePipeline(folder)
    tool = RemoveIndexTool(pipeline, guard)

    result = tool.run(str(tmp_path / "photos" / "x" / "nested"))
    assert result["count"] == 0
    assert "没有" in result["summary"]


def test_remove_missing_only(tmp_path):
    guard, folder = _guard(tmp_path)
    pipeline = _RemovePipeline(folder)
    tool = RemoveIndexTool(pipeline, guard)

    result = tool.run(missing_only=True)
    assert result["count"] == 3
    assert result["deleted"] is False


def test_remove_requires_target(tmp_path):
    guard, _ = _guard(tmp_path)
    tool = RemoveIndexTool(_RemovePipeline(tmp_path), guard)
    result = tool.run()
    assert result["count"] == 0
    assert "folder_path" in result["summary"]
