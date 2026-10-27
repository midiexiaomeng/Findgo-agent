"""兼容层：旧的 PathGuard 名字保留，内部已经是分域沙箱。

新代码请直接用 ``Sandbox``：

    from ..utils.sandbox import Sandbox

    sandbox.check(path, "read")    # 读图
    sandbox.check(path, "write")   # 写文件
    sandbox.check(path, "delete")  # 删文件
"""

from __future__ import annotations

from .sandbox import PROJECT_ROOT, Sandbox, SandboxError

#: 旧名字，等价于只读沙箱（allowed_roots 作为 image_roots）
PathSecurityError = SandboxError


class PathGuard(Sandbox):
    """历史接口：``allowed_roots`` 当作只读的 image_roots。"""

    def __init__(self, allowed_roots=None, forbidden_roots=None):
        super().__init__(
            image_roots=allowed_roots,
            workspace_roots=None,
            forbidden_roots=forbidden_roots,
        )


__all__ = [
    "PROJECT_ROOT",
    "PathGuard",
    "PathSecurityError",
    "Sandbox",
    "SandboxError",
]
