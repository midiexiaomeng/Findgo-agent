"""权限分域沙箱。

把路径分成三类，按操作类型管控：

1. ``image_roots``  用户素材目录 —— 只读，禁止写入与删除
2. ``workspace_roots`` 缓存 / 向量库 / 导出目录 —— 读写删都可以
3. 系统目录 —— 任何操作一律拒绝

每次校验都要说明这次是读、写还是删：

    sandbox.check(path, "read")     # 读图，image_roots 与 workspace 都行
    sandbox.check(path, "write")    # 写文件，只允许 workspace
    sandbox.check(path, "delete")   # 删文件，只允许 workspace
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("findgo.sandbox")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 系统敏感目录，任何模式都拒绝
_FORBIDDEN = [
    Path("C:/Windows"),
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
    Path("C:/ProgramData"),
    Path("C:/System Volume Information"),
    Path("/etc"),
    Path("/usr"),
    Path("/bin"),
]


class SandboxError(ValueError):
    """路径越过允许范围，或操作类型不被该目录允许。"""


class Sandbox:
    READ = "read"
    WRITE = "write"
    DELETE = "delete"

    def __init__(
        self,
        image_roots: list[str] | None = None,
        workspace_roots: list[str] | None = None,
        forbidden_roots: list[str] | None = None,
    ):
        image = image_roots or [
            str(PROJECT_ROOT / "datasets"),
            str(Path.home() / "Pictures"),
        ]
        workspace = workspace_roots or [str(PROJECT_ROOT / "data")]

        self.image_roots = self._resolve_all(image)
        self.workspace_roots = self._resolve_all(workspace)
        self.forbidden = self._resolve_all(
            _FORBIDDEN + [Path(p) for p in (forbidden_roots or [])]
        )

    @staticmethod
    def _resolve_all(items) -> list[Path]:
        out: list[Path] = []
        for item in items:
            try:
                out.append(Path(item).expanduser().resolve())
            except OSError:
                continue
        return out

    # ── 校验 ────────────────────────────────────────────────────────────────

    def check(self, raw_path: str | Path, mode: str = READ) -> Path:
        """校验路径，返回真实路径；不合法时抛 SandboxError。"""
        if raw_path is None or str(raw_path).strip() == "":
            raise SandboxError("路径为空")

        mode = (mode or self.READ).lower()
        if mode not in (self.READ, self.WRITE, self.DELETE):
            raise SandboxError(f"未知操作类型: {mode}")

        text = str(raw_path).strip().strip('"').strip("'")
        candidate = Path(text).expanduser()

        # 符号链接一律拒绝，避免链接指向白名单外的位置绕过边界
        if self._has_symlink(candidate):
            raise SandboxError(f"拒绝访问符号链接: {candidate}")

        try:
            resolved = candidate.resolve(strict=False)
        except OSError as e:
            raise SandboxError(f"路径无法解析: {e}") from e

        # 1. 系统目录，任何模式都拒绝
        for bad in self.forbidden:
            if self._within(resolved, bad):
                raise SandboxError(f"禁止访问系统目录: {resolved}")

        # 2. 写 / 删：只允许 workspace
        if mode in (self.WRITE, self.DELETE):
            for root in self.workspace_roots:
                if self._within(resolved, root):
                    return resolved
            raise SandboxError(
                f"{mode} 操作只允许在 workspace 内（{', '.join(str(r) for r in self.workspace_roots)}），"
                f"拒绝：{resolved}"
            )

        # 3. 读：image_roots 与 workspace 都可以
        for root in self.image_roots + self.workspace_roots:
            if self._within(resolved, root):
                return resolved

        raise SandboxError(f"路径超出允许范围: {resolved}")

    @staticmethod
    def _has_symlink(path: Path) -> bool:
        """检查路径本身或任意一级父目录是否为符号链接。"""
        try:
            current = path
            while True:
                if current.is_symlink():
                    return True
                parent = current.parent
                if parent == current:
                    break
                current = parent
        except OSError:
            return False
        return False

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            return path == root or root in path.parents
        except OSError:
            return False

    def is_safe(self, raw_path: str | Path, mode: str = READ) -> bool:
        try:
            self.check(raw_path, mode)
            return True
        except SandboxError:
            return False

    # ── 展示 ────────────────────────────────────────────────────────────────

    def describe(self) -> str:
        lines = [
            "只读目录（image_roots）：",
            *[f"  - {r}" for r in self.image_roots],
            "可写目录（workspace）：",
            *[f"  - {r}" for r in self.workspace_roots],
            "禁止目录：",
            *[f"  - {r}" for r in self.forbidden[:4]],
        ]
        return "\n".join(lines)
