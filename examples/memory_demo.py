"""结构化事实与三层记忆演示。

运行前设置环境变量 DEEPSEEK_API_KEY（摘要/事实抽取需要）。

    python examples/memory_demo.py

演示：
1. 多轮对话写入短期记忆
2. 每轮结束后 LLM 抽取结构化事实 patch
3. 展示事实如何被修改、删除
4. 清空本次任务的约束
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from findgo.agent.memory import Memory


def show(memory: Memory, title: str) -> None:
    facts = memory.get_facts()
    print(f"\n[{title}]")
    print("  结构化事实:", json.dumps(facts, ensure_ascii=False) if facts else "(空)")


def main() -> None:
    memory = Memory(persist_path="./data/memory_demo.sqlite")
    memory.new_session()

    print("=" * 60)
    print("模拟一次图片检索任务的事实演进")

    memory.patch_facts({"subject": "图片检索对象", "count": 2, "filters": {"color": "白色"}})
    show(memory, "初始约束")

    memory.patch_facts({"count": 3})
    show(memory, "用户改成 3 张")

    memory.patch_facts({"filters": {"style": "非卡通"}})
    show(memory, "追加风格约束")

    memory.patch_facts({"filters": {"color": None}})
    show(memory, "去掉颜色限制")

    memory.set_summary("用户要 3 张非卡通图片，之前要求白色但已取消。")
    print("\n[会话摘要]")
    print(" ", memory.get_summary())

    memory.add("user", "找三张非卡通的图片")
    memory.add("assistant", "已找到 3 张。")
    print("\n[短期消息]")
    for item in memory.load_messages(memory.session_id):
        print(f"  {item['role']}: {item['content']}")

    memory.clear_facts()
    show(memory, "清空本次任务约束")

    print("\n" + "=" * 60)
    print("长期记忆（跨会话偏好）请用 LongTermMemory：")
    print("  from findgo.long_term import LongTermConfig, LongTermMemory")
    print("  lt = LongTermMemory()")
    print("  lt.add(user_id, session_id, text, vector, tags)")
    print("  lt.search(user_id, vector)")
    print("  lt.clear(user_id)")


if __name__ == "__main__":
    main()
