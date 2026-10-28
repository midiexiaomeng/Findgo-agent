"""缩略图缓存：把任意尺寸图片裁成固定正方形，供对话气泡展示。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def cache_dir() -> Path:
    """data/thumbs 位于项目根目录。"""
    return Path(__file__).resolve().parents[3] / "data" / "thumbs"


def thumb_path(path: str, size: int = 112) -> str:
    """生成并缓存正方形缩略图，失败返回空字符串。"""
    from PIL import Image, ImageOps

    source = Path(path)
    try:
        stat = source.stat()
    except OSError:
        return ""

    key = hashlib.md5(
        f"{source.resolve()}|{stat.st_mtime}|{size}".encode("utf-8")
    ).hexdigest()
    target_dir = cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{key}.jpg"
    if target.exists():
        return str(target)

    try:
        with Image.open(source) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = ImageOps.fit(im, (size, size), method=Image.LANCZOS)
            im.save(target, "JPEG", quality=85)
    except Exception:
        return ""
    return str(target)
