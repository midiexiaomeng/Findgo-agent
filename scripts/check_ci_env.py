"""本地模拟 CI 环境跑测试。

CI 的 ubuntu runner 不装 torch / transformers / PyQt5 / gradio，
测试靠 importorskip 跳过 GUI 与模型相关用例。本地环境有这些包，
直接跑 pytest 覆盖不到「缺依赖」的分支，只能等 CI 报错。

用法：
    python scripts/check_ci_env.py            # 跑全部测试
    python scripts/check_ci_env.py -k diary   # 只跑匹配的用例
    python scripts/check_ci_env.py -v         # 详细输出

这是开发辅助脚本，不参与打包，也不进 CI。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: CI 环境里不存在的包，用 find_spec 拦截
BLOCKED = ("torch", "torchvision", "transformers", "gradio", "PyQt5")


class _Blocker:
    """在 import 阶段直接抛错，模拟包未安装。"""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"模拟 CI 环境：{name} 不可用")
        return None


def install_blocker() -> None:
    # 只插一次，避免重复运行时叠加
    if not any(isinstance(f, _Blocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _Blocker())


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])

    install_blocker()
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        import pytest
    except ImportError:
        print("需要先安装 pytest：pip install pytest")
        return 2

    print(f"模拟 CI 环境：屏蔽 {', '.join(BLOCKED)}")
    print()

    return pytest.main(["tests", "-q", "--no-header", *args])


if __name__ == "__main__":
    raise SystemExit(main())
