"""路径安全校验。

防止用户通过 ``../`` 或绝对路径访问系统目录：
1. 先 ``resolve()`` 把 ``..`` 和符号链接展开成真实路径
2. 命中禁止目录（Windows / Program Files / ProgramData 等）直接拒绝
3. 必须落在允许的根目录内（默认项目目录与用户主目录）
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("imgsearch_agent.path_guard")

# utils/path_guard.py → 上三层是项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 系统敏感目录，任何情况下都不允许访问
_FORBIDDEN = [
    Path("C:/Windows"),
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
    Path("C:/ProgramData"),
    Path("C:/System Volume Information"),
]


class PathSecurityError(ValueError):
    """路径越过安全边界。"""


class PathGuard:
    def __init__(
        self,
        allowed_roots: list[str] | None = None,
        forbidden_roots: list[str] | None = None,
    ):
        roots = allowed_roots or [str(PROJECT_ROOT), str(Path.home())]
        self.allowed = [Path(r).expanduser().resolve() for r in roots if r]
        self.forbidden = [Path(p).resolve() for p in _FORBIDDEN]
        for extra in forbidden_roots or []:
            try:
                self.forbidden.append(Path(extra).expanduser().resolve())
            except OSError:
                continue

    def check(self, raw_path: str | Path) -> Path:
        """校验并返回真实路径，越界时抛 PathSecurityError。"""
        if raw_path is None or str(raw_path).strip() == "":
            raise PathSecurityError("路径为空")

        text = str(raw_path).strip().strip('"').strip("'")
        candidate = Path(text).expanduser()

        try:
            resolved = candidate.resolve(strict=False)
        except OSError as e:
            raise PathSecurityError(f"路径无法解析: {e}") from e

        # 1. 禁止系统目录
        for bad in self.forbidden:
            try:
                if resolved == bad or bad in resolved.parents:
                    raise PathSecurityError(f"禁止访问系统目录: {resolved}")
            except OSError:
                continue

        # 2. 必须位于允许的根目录内
        for root in self.allowed:
            try:
                if resolved == root or root in resolved.parents:
                    return resolved
            except OSError:
                continue

        raise PathSecurityError(f"路径超出允许范围: {resolved}")

    def is_safe(self, raw_path: str | Path) -> bool:
        try:
            self.check(raw_path)
            return True
        except PathSecurityError:
            return False
