"""三层记忆引擎测试。"""

from __future__ import annotations

from imgsearch_agent.memory_engine import MemoryConfig, MemoryEngine, SessionStore


def make_engine(tmp_path, enabled=False):
    return MemoryEngine(
        config=MemoryConfig(
            db_path=str(tmp_path / "memory.db"),
            memory_enabled=enabled,
        )
    )


def test_session_store_basic(tmp_path):
    store = SessionStore(tmp_path / "m.db")
    store.append("u1", "s1", "user", "你好")
    store.append("u1", "s1", "assistant", "在的")
    assert store.count("s1") == 2
    assert store.all_messages("s1")[0]["content"] == "你好"


def test_session_isolation(tmp_path):
    store = SessionStore(tmp_path / "m.db")
    store.append("u1", "s1", "user", "会话1")
    store.append("u2", "s2", "user", "会话2")
    assert store.count("s1") == 1
    assert store.count("s2") == 1
    assert store.all_messages("s1")[0]["content"] == "会话1"


def test_summary_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "m.db")
    store.set_summary("s1", "用户偏好暗色调")
    assert store.get_summary("s1") == "用户偏好暗色调"


def test_build_messages_includes_summary(tmp_path):
    engine = make_engine(tmp_path)
    engine.add_message("u1", "s1", "user", "之前聊过猫")
    engine.sessions.set_summary("s1", "用户喜欢猫的图片")

    messages = engine.build_messages("u1", "s1", "你是助手", "找狗")
    assert messages[0]["role"] == "system"
    assert "用户喜欢猫的图片" in messages[0]["content"]
    assert messages[-1]["content"] == "找狗"


def test_build_messages_without_summary(tmp_path):
    engine = make_engine(tmp_path)
    messages = engine.build_messages("u1", "s1", "你是助手", "你好")
    assert messages[0]["content"] == "你是助手"


def test_extract_facts_parses_json(tmp_path):
    engine = make_engine(tmp_path)

    class FakeLLM:
        def chat(self, messages, **kwargs):
            return '{"useful": true, "facts": ["用户偏好暗色"], "tags": ["preference"]}'

    engine._llm = FakeLLM()
    result = engine.extract_facts("我喜欢暗色", "好的")
    assert result["useful"] is True
    assert result["facts"] == ["用户偏好暗色"]
    assert result["tags"] == ["preference"]


def test_extract_facts_handles_bad_json(tmp_path):
    engine = make_engine(tmp_path)

    class FakeLLM:
        def chat(self, messages, **kwargs):
            return "不是 JSON"

    engine._llm = FakeLLM()
    result = engine.extract_facts("随便聊聊", "哈哈")
    assert result["useful"] is False
    assert result["facts"] == []


def test_memory_disabled_skips_longterm(tmp_path):
    engine = make_engine(tmp_path, enabled=False)

    class FakeLLM:
        def chat(self, messages, **kwargs):
            return '{"useful": true, "facts": ["x"], "tags": []}'

    engine._llm = FakeLLM()
    result = engine.after_turn("u1", "s1", "我喜欢猫", "记住了")
    assert result["facts_saved"] == 0


def test_set_enabled_toggles(tmp_path):
    engine = make_engine(tmp_path)
    engine.set_enabled(False)
    assert engine.config.memory_enabled is False
    engine.set_enabled(True)
    assert engine.config.memory_enabled is True
