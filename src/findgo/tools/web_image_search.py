"""联网图片搜索与下载工具。

流程：搜索图片 URL → 下载到缓存区 → 自动建索引 → 返回本地路径。
下载的图和用户原图分开放，缓存有独立容量上限。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base_tool import BaseTool

logger = logging.getLogger("findgo.web_image_search")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class WebImageSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="要搜索的图片内容")
    top_k: int = Field(20, ge=1, le=200, description="下载数量")
    index: bool = Field(True, description="下载后是否立即建立索引")


class WebImageSearchTool(BaseTool):
    name = "web_image_search"
    description = "联网搜索图片并下载到本地缓存，可选自动建索引，用于本地图库不够时补充"
    permission = "file_write"
    timeout = 600.0
    args_schema = WebImageSearchArgs

    def __init__(self, pipeline=None, cache=None):
        self._pipeline = pipeline
        self._cache = cache

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    @property
    def cache(self):
        if self._cache is None:
            from ..config import AgentConfig
            from ..downloads import DownloadCache
            from ..retriever.runtime import PROJECT_ROOT

            cfg = AgentConfig()
            root = Path(cfg.download_path).expanduser()
            if not root.is_absolute():
                root = Path(PROJECT_ROOT) / root
            self._cache = DownloadCache(root, cfg.download_max_gb)
        return self._cache

    def run(
        self, query: str, top_k: int = 20, index: bool = True
    ) -> dict[str, Any]:
        urls = self._search_urls(query, top_k)
        if not urls:
            return {
                "downloaded": 0,
                "paths": [],
                "summary": "联网搜索没有拿到图片链接，可能是网络问题或被限流",
            }

        saved: list[str] = []
        failed = 0

        for url in urls:
            if len(saved) >= top_k:
                break
            try:
                target = self._download(url)
                if target is not None:
                    saved.append(str(target))
            except Exception as e:
                logger.debug("下载失败 %s: %s", url, e)
                failed += 1

        self.cache.enforce_limit()

        indexed = 0
        if index and saved:
            try:
                stats = self.pipeline.index_folder(
                    self.cache.root, recursive=False, skip_existing=True
                )
                indexed = stats.indexed
            except Exception as e:
                logger.warning("下载图片建索引失败: %s", e)

        return {
            "downloaded": len(saved),
            "indexed": indexed,
            "failed": failed,
            "paths": saved,
            "cache_dir": str(self.cache.root),
            "summary": (
                f"联网下载 {len(saved)} 张"
                + (f"，新索引 {indexed} 张" if indexed else "")
                + (f"，失败 {failed} 张" if failed else "")
            ),
        }

    # ── 内部 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _search_urls(query: str, count: int) -> list[str]:
        import requests

        response = requests.get(
            "https://www.bing.com/images/search",
            params={"q": query, "form": "HDRSC2", "first": 1},
            headers={"User-Agent": _UA},
            timeout=15,
        )
        response.raise_for_status()

        found = re.findall(r'murl&quot;:&quot;(https?://[^&]+?)&quot;', response.text)
        unique = list(dict.fromkeys(found))
        return unique[: max(count * 2, 10)]

    def _download(self, url: str) -> Path | None:
        import requests

        response = requests.get(url, headers={"User-Agent": _UA}, timeout=12)
        if response.status_code != 200 or len(response.content) < 2048:
            return None

        suffix = Path(url.split("?")[0]).suffix.lower().lstrip(".")
        if suffix not in ("jpg", "jpeg", "png", "webp", "bmp"):
            suffix = "jpg"

        return self.cache.save(url, response.content, suffix)
