"""联网图片下载缓存。

和暂存区（trash_staging）完全分开：
- 暂存区存的是「待删除」的原图引用
- 这里存的是「网上拉下来、还没索引或已索引」的新图

有独立容量上限，超出按最旧文件淘汰。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger("imgsearch_agent.downloads")


class DownloadCache:
    def __init__(self, root: str | Path, max_gb: float = 5.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(1, int(float(max_gb) * 1024**3))

    def save(self, url: str, data: bytes, ext: str = "jpg") -> Path:
        key = hashlib.md5(url.encode("utf-8")).hexdigest()[:20]
        target = self.root / f"{key}.{ext.lstrip('.')}"
        target.write_bytes(data)
        return target

    def enforce_limit(self) -> int:
        """超出容量时从最旧的文件开始删，返回删除数量。"""
        files = [p for p in self.root.glob("*") if p.is_file()]
        if not files:
            return 0

        files.sort(key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        removed = 0

        for path in files:
            if total <= self.max_bytes:
                break
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
                removed += 1
            except OSError:
                continue

        if removed:
            logger.info("下载缓存超出上限，淘汰 %d 个文件", removed)
        return removed

    def stats(self) -> dict:
        files = [p for p in self.root.glob("*") if p.is_file()]
        return {
            "count": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "max_bytes": self.max_bytes,
        }

    def clear(self) -> int:
        files = [p for p in self.root.glob("*") if p.is_file()]
        for path in files:
            try:
                path.unlink()
            except OSError:
                continue
        return len(files)
