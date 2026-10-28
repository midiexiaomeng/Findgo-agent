"""三层记忆引擎演示。

运行前设置环境变量 DEEPSEEK_API_KEY（用于摘要与事实抽取）。

    python examples/memory_demo.py

演示内容：
1. 多轮对话写入短期记忆
2. 每轮结束后自动抽取事实写入 Qdrant 长期记忆
3. 按语义检索历史记忆
4. 清空该用户的全部长期记忆

开关说明：
- 关闭长期记忆：engine.set_enabled(False)，之后不再写入也不再检索
- 清空记忆：engine.clear_memories(user_id)
- 删除单条：engine.delete_memory(memory_id)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgsearch_agent.memory_engine import MemoryConfig, MemoryEngine


def main() -> None:
    engine = MemoryEngine(
        config=MemoryConfig(db_path="./data/memory_demo.db")
    )

    user_id = "demo_user"
    session_id = "demo_session"

    turns = [
        ("我经常找猫的图片", "好的，记住你偏好猫的图片。"),
        ("我更喜欢暗色调的照片", "明白，之后优先暗色调。"),
        ("今天天气不错", "嗯，适合出门走走。"),
        ("帮我找两张海边的日落", "已找到两张海边的日落。"),
    ]

    print("=" * 60)
    for user_text, reply in turns:
        engine.add_message(user_id, session_id, "user", user_text)
        engine.add_message(user_id, session_id, "assistant", reply)
        result = engine.after_turn(user_id, session_id, user_text, reply)
        print(f"[轮次] {user_text}")
        print(
            f"  长期记忆 +{result['facts_saved']} 条"
            f" | 摘要压缩：{'是' if result['summarized'] else '否'}"
        )

    print("\n" + "=" * 60)
    print("[长期记忆清单]")
    memories = engine.list_memories(user_id)
    if not memories:
        print("  （空，可能是没有配置 API Key 或本轮没有值得保存的事实）")
    for item in memories:
        print(f"  - {item['text']}   [{item['tags']}]")

    print("\n" + "=" * 60)
    print("[语义检索] 查询：我喜欢什么色调")
    for hit in engine.search(user_id, "我喜欢什么色调"):
        print(f"  - {hit['text']}   相似度 {hit['score']}")

    print("\n" + "=" * 60)
    print("[组装 prompt]")
    messages = engine.build_messages(user_id, session_id, "你是图像检索助手", "找几张图")
    print(f"  系统提示长度：{len(messages[0]['content'])} 字")
    print(f"  消息条数：{len(messages)}")

    print("\n" + "=" * 60)
    print("[清空长期记忆]")
    removed = engine.clear_memories(user_id)
    print(f"  已删除 {removed} 条，剩余 {len(engine.list_memories(user_id))} 条")


if __name__ == "__main__":
    main()
