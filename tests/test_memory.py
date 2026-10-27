"""Memory 测试：内存模式、持久化重载、轨迹记录。"""

from __future__ import annotations

from imgsearch_agent.agent.memory import Memory


def test_memory_in_process():
    m = Memory()
    m.add("user", "找猫")
    m.add("assistant", "好的")
    assert m.recent(2) == [
        {"role": "user", "content": "找猫"},
        {"role": "assistant", "content": "好的"},
    ]
    assert len(m) == 2


def test_memory_persist_and_reload(tmp_path):
    db = tmp_path / "sessions.sqlite"
    m = Memory(session_id="s1", persist_path=db)
    m.add("user", "找海边的日落")
    m.close()

    assert db.exists()
    m2 = Memory(session_id="s1", persist_path=db)
    assert m2.recent(1)[0]["content"] == "找海边的日落"
    m2.close()


def test_trace_records_tool_call(tmp_path):
    db = tmp_path / "sessions.sqlite"
    m = Memory(session_id="s1", persist_path=db)
    m.add_trace(
        "t1",
        "tool_call",
        "image_retrieval",
        {"query": "猫"},
        {"images": [{"path": "a.jpg"}]},
    )
    traces = m.traces()
    assert traces[0]["tool_name"] == "image_retrieval"
    assert traces[0]["tool_args"]["query"] == "猫"
    m.close()


def test_memory_mode_does_not_create_db(tmp_path):
    db = tmp_path / "sessions.sqlite"
    m = Memory()
    m.add("user", "hi")
    assert not db.exists()
    m.close()
