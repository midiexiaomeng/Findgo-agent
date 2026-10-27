"""暂存区：Agent 的“删除”永远先落到这里，不直接动用户原图。

规则：

1. 暂存时用硬链接（同分区）或移动（跨分区），不复制副本
2. 每一步写 manifest.db，记录原始路径、md5、时间、任务 ID
3. 恢复 = 删暂存链接，原图原样保留
4. 永久删除 = 删原图 + 删暂存链接，只在用户确认或过期时发生
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("imgsearch_agent.trash")


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class TrashStaging:
    STATUS_STAGED = "staged"
    STATUS_PURGED = "purged"
    STATUS_RESTORED = "restored"

    def __init__(self, root: str | Path, db_path: str | Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.root / "manifest.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── 存储 ────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS staged (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    original_abs_path TEXT NOT NULL,
                    staged_path TEXT NOT NULL,
                    file_md5 TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    staged_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'staged',
                    is_hardlink INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON staged(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task ON staged(task_id)"
            )

    # ── 暂存 ────────────────────────────────────────────────────────────────

    def stage(self, paths: list[str | Path], task_id: str) -> dict:
        """把图片放进暂存区，返回统计。原图在硬链接模式下保持不动。"""
        staged = 0
        skipped = 0
        failed: list[str] = []
        details: list[dict] = []

        for raw in paths:
            source = Path(raw)
            if not source.is_file():
                failed.append(f"{source} 不是文件")
                continue

            try:
                md5 = file_md5(source)
                size = source.stat().st_size
            except OSError as e:
                failed.append(f"{source} 读取失败: {e}")
                continue

            # 内容相同但路径不同的图片要各自留一条记录
            path_hash = hashlib.md5(f"{md5}|{source}".encode("utf-8")).hexdigest()[:16]
            row_id = f"{task_id}:{path_hash}"
            with self._connect() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM staged WHERE id = ? AND status = ?",
                    (row_id, self.STATUS_STAGED),
                ).fetchone()
            if exists:
                skipped += 1
                continue

            bucket = self.root / task_id / md5[:2]
            bucket.mkdir(parents=True, exist_ok=True)
            target = bucket / f"{md5}_{source.name}"

            is_hardlink = 1
            try:
                if target.exists():
                    target.unlink()
                os.link(source, target)
            except OSError:
                # 跨分区或文件系统不支持硬链接，退化为移动（单份，不复制）
                is_hardlink = 0
                try:
                    shutil.move(str(source), str(target))
                except OSError as e:
                    failed.append(f"{source} 移动失败: {e}")
                    continue

            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO staged "
                    "(id, task_id, original_abs_path, staged_path, file_md5, size, "
                    " staged_at, status, is_hardlink) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row_id,
                        task_id,
                        str(source),
                        str(target),
                        md5,
                        size,
                        time.time(),
                        self.STATUS_STAGED,
                        is_hardlink,
                    ),
                )

            staged += 1
            details.append(
                {
                    "id": row_id,
                    "original": str(source),
                    "staged": str(target),
                    "hardlink": bool(is_hardlink),
                }
            )

        return {
            "task_id": task_id,
            "staged": staged,
            "skipped": skipped,
            "failed": failed,
            "details": details,
        }

    # ── 恢复 ────────────────────────────────────────────────────────────────

    def restore(self, task_id: str | None = None, ids: list[str] | None = None) -> dict:
        rows = self._select(self.STATUS_STAGED, task_id, ids)
        restored = 0
        conflicts = 0
        failed: list[str] = []
        restored_paths: list[str] = []

        for row in rows:
            original = Path(row["original_abs_path"])
            staged = Path(row["staged_path"])

            if row["is_hardlink"]:
                # 原图还在，删掉暂存链接即可
                try:
                    if staged.exists():
                        staged.unlink()
                except OSError as e:
                    failed.append(f"{staged} 删除失败: {e}")
                    continue
            else:
                # 跨分区是移动，需要移回原位置
                target = original
                if target.exists():
                    conflicts += 1
                    target = target.with_name(
                        f"{target.stem}_bak_{int(time.time())}{target.suffix}"
                    )
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staged), str(target))
                except OSError as e:
                    failed.append(f"{staged} 恢复失败: {e}")
                    continue

            with self._connect() as conn:
                conn.execute(
                    "UPDATE staged SET status = ? WHERE id = ?",
                    (self.STATUS_RESTORED, row["id"]),
                )
            restored += 1
            restored_paths.append(str(original))

        return {
            "restored": restored,
            "conflicts": conflicts,
            "failed": failed,
            "paths": restored_paths,
        }

    # ── 永久删除 ────────────────────────────────────────────────────────────

    def purge(self, task_id: str | None = None, ids: list[str] | None = None) -> dict:
        rows = self._select(self.STATUS_STAGED, task_id, ids)
        purged = 0
        freed = 0
        failed: list[str] = []
        purged_paths: list[str] = []

        for row in rows:
            original = Path(row["original_abs_path"])
            staged = Path(row["staged_path"])

            for target in (original, staged):
                try:
                    if target.exists():
                        target.unlink()
                except OSError as e:
                    failed.append(f"{target} 删除失败: {e}")

            with self._connect() as conn:
                conn.execute(
                    "UPDATE staged SET status = ? WHERE id = ?",
                    (self.STATUS_PURGED, row["id"]),
                )
            purged += 1
            freed += int(row["size"])
            purged_paths.append(str(original))

        return {
            "purged": purged,
            "freed_bytes": freed,
            "failed": failed,
            "paths": purged_paths,
        }

    # ── 清理 ────────────────────────────────────────────────────────────────

    def cleanup(self, max_days: int = 14, max_total_gb: float = 10.0) -> dict:
        """先按 TTL 清，再按容量从旧到新清。"""
        now = time.time()
        cutoff = now - max_days * 86400

        with self._connect() as conn:
            expired = conn.execute(
                "SELECT id FROM staged WHERE status = ? AND staged_at < ?",
                (self.STATUS_STAGED, cutoff),
            ).fetchall()

        ttl_ids = [row["id"] for row in expired]
        ttl_result = self.purge(ids=ttl_ids) if ttl_ids else {"purged": 0, "freed_bytes": 0}

        limit_bytes = int(max_total_gb * 1024**3)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, size FROM staged WHERE status = ? ORDER BY staged_at ASC",
                (self.STATUS_STAGED,),
            ).fetchall()

        total = sum(int(r["size"]) for r in rows)
        over_ids: list[str] = []
        for row in rows:
            if total <= limit_bytes:
                break
            over_ids.append(row["id"])
            total -= int(row["size"])

        over_result = self.purge(ids=over_ids) if over_ids else {"purged": 0, "freed_bytes": 0}

        return {
            "expired_purged": ttl_result.get("purged", 0),
            "size_purged": over_result.get("purged", 0),
            "freed_bytes": ttl_result.get("freed_bytes", 0)
            + over_result.get("freed_bytes", 0),
            "total_purged": ttl_result.get("purged", 0) + over_result.get("purged", 0),
        }

    # ── 查询 ────────────────────────────────────────────────────────────────

    def list_staged(self, task_id: str | None = None) -> list[dict]:
        rows = self._select(self.STATUS_STAGED, task_id, None)
        return [
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "original": row["original_abs_path"],
                "staged": row["staged_path"],
                "size": row["size"],
                "staged_at": row["staged_at"],
                "hardlink": bool(row["is_hardlink"]),
            }
            for row in rows
        ]

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS total "
                "FROM staged WHERE status = ?",
                (self.STATUS_STAGED,),
            ).fetchone()
        return {"count": int(row["n"]), "total_bytes": int(row["total"])}

    def _select(
        self,
        status: str,
        task_id: str | None,
        ids: list[str] | None,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM staged WHERE status = ?"
        params: list = [status]
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        if ids:
            placeholders = ",".join("?" for _ in ids)
            query += f" AND id IN ({placeholders})"
            params.extend(ids)
        with self._connect() as conn:
            return conn.execute(query, params).fetchall()
