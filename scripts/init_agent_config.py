"""初始化本地 Agent 配置：从模板复制一份可填 Key 的 agent.yaml。

    python scripts/init_agent_config.py

模板在 configs/agent.example.yaml，随包分发；
生成 configs/agent.yaml 后不会被 git 跟踪，也不会进发布包。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "configs" / "agent.yaml"
EXAMPLE = ROOT / "configs" / "agent.example.yaml"


def main() -> int:
    if LOCAL.exists():
        print(f"已存在，跳过: {LOCAL}")
        return 0
    if not EXAMPLE.exists():
        print(f"模板不存在: {EXAMPLE}")
        return 1
    LOCAL.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"已创建: {LOCAL}")
    print("请编辑该文件填入 api_key，或用环境变量 DEEPSEEK_API_KEY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
