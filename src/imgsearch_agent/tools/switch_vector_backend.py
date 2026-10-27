"""向量后端切换工具。"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.switch_backend")


class SwitchBackendArgs(BaseModel):
    backend_name: str = Field(..., description="目标向量库：chroma 或 qdrant")


class SwitchVectorBackendTool(BaseTool):
    name = "switch_vector_backend"
    description = "切换向量库后端（chroma / qdrant）；两个后端数据不共享，切换后需重新索引"
    args_schema = SwitchBackendArgs

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(self, backend_name: str) -> dict[str, Any]:
        name = (backend_name or "").strip().lower()
        if name not in {"chroma", "qdrant"}:
            return {
                "backend": self.pipeline.store.backend_name,
                "summary": f"不支持的向量库：{backend_name}，可选 chroma 或 qdrant",
                "error": "unsupported backend",
            }

        try:
            message = self.pipeline.switch_backend(name)
        except Exception as e:
            logger.warning("切换向量库失败: %s", e)
            return {
                "backend": self.pipeline.store.backend_name,
                "summary": f"切换失败：{e}",
                "error": str(e),
            }

        return {
            "backend": self.pipeline.store.backend_name,
            "count": self.pipeline.store.count(),
            "summary": message,
        }
