"""重复图片检测：按向量相似度找出视觉重复的分组。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathSecurityError, PathGuard
from .base_tool import BaseTool

logger = logging.getLogger("findgo.find_duplicates")


class FindDuplicatesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str = Field("", description="只在该文件夹内查找，空表示全库")
    threshold: float = Field(0.97, ge=0.5, le=1.0, description="相似度阈值")
    max_items: int = Field(5000, ge=1, le=50000, description="最多分析多少张")


class FindDuplicatesTool(BaseTool):
    name = "find_duplicates"
    description = "在索引里找出视觉重复的图片，返回相似度超过阈值的分组"
    args_schema = FindDuplicatesArgs

    def __init__(self, pipeline=None, guard: PathGuard | None = None):
        self._pipeline = pipeline
        self.guard = guard or PathGuard()

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(
        self,
        folder_path: str = "",
        threshold: float = 0.97,
        max_items: int = 5000,
    ) -> dict[str, Any]:
        records = self.pipeline.store.get_vectors(limit=max_items)
        if not records:
            return {"groups": [], "count": 0, "total_images": 0, "summary": "索引为空"}

        if folder_path:
            try:
                scope = str(self.guard.check(folder_path, "read")).lower()
            except PathSecurityError as e:
                return {
                    "groups": [],
                    "count": 0,
                    "total_images": 0,
                    "summary": f"路径不安全，已拒绝：{e}",
                }
            records = [
                r
                for r in records
                if str(r[2].get("path", "")).lower().startswith(scope)
            ]

        if len(records) < 2:
            return {
                "groups": [],
                "count": 0,
                "total_images": len(records),
                "summary": "候选少于 2 张，无需比较",
            }

        paths = [str(r[2].get("path", "")) for r in records]
        matrix = np.stack([r[1] for r in records])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-8)
        sims = matrix @ matrix.T

        parent = list(range(len(records)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in np.argwhere(np.triu(sims, 1) >= float(threshold)):
            union(int(i), int(j))

        buckets: dict[int, list[int]] = {}
        for i in range(len(records)):
            buckets.setdefault(find(i), []).append(i)

        groups = []
        for members in buckets.values():
            if len(members) < 2:
                continue
            best = 0.0
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    best = max(best, float(sims[members[a]][members[b]]))
            groups.append(
                {
                    "count": len(members),
                    "max_similarity": round(best, 4),
                    "paths": [paths[i] for i in members],
                }
            )

        groups.sort(key=lambda g: (-g["count"], -g["max_similarity"]))
        return {
            "groups": groups[:50],
            "count": len(groups),
            "total_images": len(records),
            "summary": f"比较 {len(records)} 张，发现 {len(groups)} 组重复",
        }
