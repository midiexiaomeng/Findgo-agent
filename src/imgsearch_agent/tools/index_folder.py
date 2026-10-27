"""目录索引工具：对指定文件夹增量构建图像索引。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.index_folder")


class IndexFolderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str = Field(..., min_length=1, description="要建立索引的文件夹路径")
    rebuild: bool = Field(False, description="True 全量重建，False 增量更新")


class IndexFolderTool(BaseTool):
    name = "index_folder"
    permission = "index_write"
    timeout = 3600.0
    description = "对指定文件夹构建图像索引；默认增量，已索引过的图片自动跳过"
    args_schema = IndexFolderArgs

    def __init__(self, pipeline=None, guard: PathGuard | None = None):
        self._pipeline = pipeline
        self.guard = guard or PathGuard()
        self._mtime: dict[str, float] = {}
        self._load_cache()

    # ── 目录修改时间缓存 ──────────────────────────────────────────────────

    @staticmethod
    def _cache_file() -> Path:
        from ..retriever.runtime import PROJECT_ROOT

        return Path(PROJECT_ROOT) / "data" / "index_mtime.json"

    def _load_cache(self) -> None:
        try:
            path = self._cache_file()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._mtime = {str(k): float(v) for k, v in data.items()}
        except Exception:
            self._mtime = {}

    def _save_cache(self) -> None:
        try:
            path = self._cache_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._mtime, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(self, folder_path: str, rebuild: bool = False) -> dict[str, Any]:
        try:
            folder = self.guard.check(folder_path, "read")
        except PathSecurityError as e:
            return {
                "count": 0,
                "total": 0,
                "failed": 0,
                "summary": f"路径不安全，已拒绝：{e}",
                "error": str(e),
            }

        if not folder.is_dir():
            return {
                "count": 0,
                "total": 0,
                "failed": 0,
                "summary": f"{folder} 不是文件夹",
            }

        key = str(folder).lower()
        try:
            mtime = folder.stat().st_mtime
        except OSError:
            mtime = 0.0

        # 目录本身没变化就直接跳过（新增/删除文件会改变 mtime）
        if not rebuild and self._mtime.get(key) == mtime:
            return {
                "count": 0,
                "total": 0,
                "failed": 0,
                "cached": True,
                "summary": f"{folder} 自上次索引后没有变化，跳过",
            }

        stats = self.pipeline.index_folder(
            folder,
            recursive=True,
            reset=rebuild,
            skip_existing=not rebuild,
        )

        self._mtime[key] = mtime
        self._save_cache()

        return {
            "count": stats.indexed,
            "total": stats.total,
            "failed": stats.failed,
            "cached": False,
            "summary": stats.summary(),
        }
