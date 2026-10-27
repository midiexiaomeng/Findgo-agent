"""检索结果导出工具：把查询结果写成 CSV / TXT 清单。"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.export_results")


class ExportResultsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="检索描述")
    top_k: int = Field(20, ge=1, le=1000, description="导出条数")
    folder_path: str = Field("", description="限定检索的文件夹路径")
    output_format: str = Field("csv", pattern="^(csv|txt)$", description="csv 或 txt")


class ExportResultsTool(BaseTool):
    name = "export_results"
    description = "把一次检索的结果导出成 CSV 或 TXT 清单，方便交给别的程序处理"
    args_schema = ExportResultsArgs

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
        query: str,
        top_k: int = 20,
        folder_path: str = "",
        output_format: str = "csv",
    ) -> dict[str, Any]:
        scope = ""
        if folder_path:
            try:
                scope = str(self.guard.check(folder_path, "read"))
            except PathSecurityError as e:
                return {"path": "", "count": 0, "summary": f"路径不安全，已拒绝：{e}"}

        # 导出不需要语义重排，按向量分排序即可，速度更快
        response = self.pipeline.searcher.search_text(
            query,
            top_k=top_k,
            scope=scope or None,
            use_rerank=False,
        )

        rows = [
            {
                "rank": item.rank,
                "path": item.path,
                "score": round(float(item.score), 4),
                "caption": item.caption or "",
            }
            for item in response.results
        ]

        from ..retriever.runtime import PROJECT_ROOT

        out_dir = Path(PROJECT_ROOT) / "data" / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "csv" if output_format.lower() == "csv" else "txt"
        target = out_dir / f"export_{stamp}.{suffix}"

        try:
            self.guard.check(target, "write")
        except PathSecurityError as e:
            return {"path": "", "count": 0, "summary": f"导出目录不在 workspace 内：{e}"}

        try:
            if suffix == "csv":
                with target.open("w", encoding="utf-8-sig", newline="") as fh:
                    writer = csv.DictWriter(
                        fh, fieldnames=["rank", "path", "score", "caption"]
                    )
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                lines = [
                    f"{row['rank']}\t{row['score']}\t{row['path']}" for row in rows
                ]
                target.write_text("\n".join(lines), encoding="utf-8")
        except OSError as e:
            logger.warning("导出失败: %s", e)
            return {"path": "", "count": 0, "summary": f"写入失败：{e}"}

        return {
            "path": str(target),
            "count": len(rows),
            "summary": f"已导出 {len(rows)} 条到 {target.name}",
        }
