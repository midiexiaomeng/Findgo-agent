"""暂存区测试：硬链接、恢复、永久删除、清理。"""

from __future__ import annotations

import os
import time
from pathlib import Path

from findgo.trash import TrashStaging


def make_image(root, name="a.jpg", size=128):
    path = root / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_stage_keeps_original_via_hardlink(tmp_path):
    src = make_image(tmp_path)
    staging = TrashStaging(tmp_path / "trash")

    result = staging.stage([str(src)], "t1")
    assert result["staged"] == 1
    assert src.exists(), "硬链接模式下原图应保留"

    items = staging.list_staged()
    assert len(items) == 1
    assert items[0]["original"] == str(src)
    assert os.path.exists(items[0]["staged"])


def test_stage_skips_duplicate(tmp_path):
    src = make_image(tmp_path)
    staging = TrashStaging(tmp_path / "trash")

    staging.stage([str(src)], "t1")
    second = staging.stage([str(src)], "t1")
    assert second["staged"] == 0
    assert second["skipped"] == 1


def test_restore_removes_link_keeps_file(tmp_path):
    src = make_image(tmp_path)
    staging = TrashStaging(tmp_path / "trash")
    staging.stage([str(src)], "t1")

    result = staging.restore(task_id="t1")
    assert result["restored"] == 1
    assert src.exists()
    assert staging.list_staged() == []


def test_purge_removes_original(tmp_path):
    src = make_image(tmp_path)
    staging = TrashStaging(tmp_path / "trash")
    staged = staging.stage([str(src)], "t1")

    result = staging.purge(task_id="t1")
    assert result["purged"] == 1
    assert not src.exists()
    assert not os.path.exists(staged["details"][0]["staged"])


def test_restore_requires_target(tmp_path):
    staging = TrashStaging(tmp_path / "trash")
    assert staging.restore()["restored"] == 0


def test_cleanup_by_ttl(tmp_path):
    src = make_image(tmp_path)
    staging = TrashStaging(tmp_path / "trash")
    staging.stage([str(src)], "t1")

    # 手动把时间拨到 30 天前
    with staging._connect() as conn:
        conn.execute(
            "UPDATE staged SET staged_at = ?", (time.time() - 30 * 86400,)
        )

    result = staging.cleanup(max_days=14, max_total_gb=100)
    assert result["expired_purged"] == 1
    assert not src.exists()


def test_stats(tmp_path):
    src = make_image(tmp_path, size=256)
    staging = TrashStaging(tmp_path / "trash")
    staging.stage([str(src)], "t1")

    stats = staging.stats()
    assert stats["count"] == 1
    assert stats["total_bytes"] == 256

def test_stage_rolls_back_when_manifest_write_fails(tmp_path, monkeypatch):
    src = make_image(tmp_path)
    staging = TrashStaging(tmp_path / "trash")

    real_connect = staging._connect
    calls = {"n": 0}

    def flaky_connect():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("db down")
        return real_connect()

    monkeypatch.setattr(staging, "_connect", flaky_connect)

    result = staging.stage([str(src)], "t1")
    assert result["staged"] == 0
    assert result["failed"]

    bucket = tmp_path / "trash" / "t1"
    leftovers = list(bucket.rglob("*.jpg")) if bucket.exists() else []
    assert leftovers == [], "写库失败后应回滚暂存链接"


def test_prune_orphans_removes_stale_rows(tmp_path):
    src = make_image(tmp_path)
    staging = TrashStaging(tmp_path / "trash")
    staging.stage([str(src)], "t1")

    items = staging.list_staged()
    Path(items[0]["staged"]).unlink()

    removed = staging.prune_orphans()
    assert removed == 1
    assert staging.list_staged() == []


def test_cleanup_reports_orphans(tmp_path):
    src = make_image(tmp_path)
    staging = TrashStaging(tmp_path / "trash")
    staging.stage([str(src)], "t1")
    Path(staging.list_staged()[0]["staged"]).unlink()

    result = staging.cleanup(max_days=14, max_total_gb=100)
    assert result["orphans_cleaned"] == 1
