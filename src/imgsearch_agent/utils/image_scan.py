"""目录图像扫描。

只负责"找出文件夹里有哪些图片"，不做编码、不写库。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("imgsearch_agent.image_scan")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".avif",
}


def scan_images(
    folder: str | Path,
    recursive: bool = True,
    limit: int = 0,
) -> list[Path]:
    """扫描目录下的图片。

    Args:
        folder: 文件夹路径（调用方应先用 PathGuard 校验）
        recursive: 是否包含子目录
        limit: 最多返回多少张，0 表示不限制
    """
    root = Path(folder)
    if not root.is_dir():
        return []

    iterator = root.rglob("*") if recursive else root.glob("*")
    found: list[Path] = []

    for path in iterator:
        try:
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            found.append(path)
            if limit and len(found) >= limit:
                break
        except OSError:
            continue

    found.sort(key=lambda p: str(p).lower())
    logger.info("扫描 %s：%d 张图片", root, len(found))
    return found
