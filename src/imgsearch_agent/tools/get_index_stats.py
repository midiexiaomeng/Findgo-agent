"""索引统计工具。"""

from __future__ import annotations

from typing import Any

from .base_tool import BaseTool


class GetIndexStatsTool(BaseTool):
    name = "get_index_stats"
    description = "查询当前索引状态：图片数量、向量库类型、设备与模型加载情况"
    args_schema = None

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(self) -> dict[str, Any]:
        stats = self.pipeline.stats()
        count = int(stats.get("images", 0))
        device = str(stats.get("device", ""))
        return {
            "count": count,
            "backend": stats["backend"],
            "collection": stats["collection"],
            "device": device,
            "encoder_loaded": bool(stats.get("encoder_loaded")),
            "summary": (
                f"{stats['backend']}（collection={stats['collection']}）"
                f"当前共 {count} 张图片，设备 {device}"
            ),
        }
