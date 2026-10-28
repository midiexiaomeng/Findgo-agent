"""向量软删除联动测试。"""

from __future__ import annotations

from pathlib import Path

from findgo.tools import TrashManageTool
from findgo.trash import TrashStaging
from findgo.utils.sandbox import Sandbox


class _Store:
    def __init__(self):
        self.updated: list[tuple[list[str], dict]] = []
        self.deleted: list[str] = []

    def update_payload(self, ids, payload):
        self.updated.append((list(ids), dict(payload)))

    def delete(self, ids):
        self.deleted.extend(ids)


class _Pipeline:
    def __init__(self):
        self.store = _Store()


def build(tmp_path, count=2):
    images = tmp_path / "images"
    images.mkdir()
    paths = []
    for i in range(count):
        p = images / f"{i}.jpg"
        p.write_bytes(b"x" * 64 + bytes([i]))
        paths.append(str(p))

    pipeline = _Pipeline()
    tool = TrashManageTool(
        pipeline=pipeline,
        guard=Sandbox(
            image_roots=[str(images)], workspace_roots=[str(tmp_path)]
        ),
        staging=TrashStaging(tmp_path / "trash"),
    )
    return tool, pipeline, paths


def test_stage_marks_vectors(tmp_path):
    tool, pipeline, paths = build(tmp_path)
    tool.run("stage", paths=paths, task_id="t1")
    assert pipeline.store.updated
    ids, payload = pipeline.store.updated[-1]
    assert len(ids) == 2
    assert payload["is_staged"] is True


def test_restore_unmarks_vectors(tmp_path):
    tool, pipeline, paths = build(tmp_path)
    tool.run("stage", paths=paths, task_id="t1")
    tool.run("restore", task_id="t1")
    _, payload = pipeline.store.updated[-1]
    assert payload["is_staged"] is False


def test_purge_deletes_vectors(tmp_path):
    tool, pipeline, paths = build(tmp_path)
    tool.run("stage", paths=paths, task_id="t1")
    tool.run("purge", task_id="t1", allow_danger=True)
    assert len(pipeline.store.deleted) == 2


def test_stage_rejects_unsafe_path(tmp_path):
    tool, pipeline, _ = build(tmp_path)
    result = tool.run("stage", paths=["C:/Windows/win.ini"], task_id="t1")
    assert result["staged"] == 0
    assert "不安全" in result["summary"]
    assert pipeline.store.updated == []
