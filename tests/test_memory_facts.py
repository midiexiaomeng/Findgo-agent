"""结构化事实与摘要存储测试。"""

from __future__ import annotations

from imgsearch_agent.agent.memory import Memory


def make_memory(tmp_path):
    memory = Memory(persist_path=tmp_path / "mem.sqlite")
    memory.new_session()
    return memory


def test_facts_initial_empty(tmp_path):
    memory = make_memory(tmp_path)
    assert memory.get_facts() == {}


def test_patch_adds_fields(tmp_path):
    memory = make_memory(tmp_path)
    memory.patch_facts({"count": 2, "filters": {"color": "白色"}})
    facts = memory.get_facts()
    assert facts["count"] == 2
    assert facts["filters"]["color"] == "白色"


def test_patch_overwrites(tmp_path):
    memory = make_memory(tmp_path)
    memory.patch_facts({"count": 2})
    memory.patch_facts({"count": 3})
    assert memory.get_facts()["count"] == 3


def test_patch_null_deletes_field(tmp_path):
    memory = make_memory(tmp_path)
    memory.patch_facts({"filters": {"color": "白色", "style": "非卡通"}})
    memory.patch_facts({"filters": {"color": None}})
    filters = memory.get_facts()["filters"]
    assert "color" not in filters
    assert filters["style"] == "非卡通"


def test_clear_facts(tmp_path):
    memory = make_memory(tmp_path)
    memory.patch_facts({"count": 5})
    memory.clear_facts()
    assert memory.get_facts() == {}


def test_facts_isolated_by_session(tmp_path):
    memory = make_memory(tmp_path)
    first = memory.session_id
    memory.patch_facts({"count": 1})

    memory.new_session()
    second = memory.session_id
    memory.patch_facts({"count": 9})

    assert memory.get_facts(first)["count"] == 1
    assert memory.get_facts(second)["count"] == 9


def test_summary_roundtrip(tmp_path):
    memory = make_memory(tmp_path)
    memory.set_summary("用户要 3 张非卡通图片")
    assert "非卡通" in memory.get_summary()


def test_summary_isolated_by_session(tmp_path):
    memory = make_memory(tmp_path)
    first = memory.session_id
    memory.set_summary("会话一摘要")

    memory.new_session()
    memory.set_summary("会话二摘要")

    assert memory.get_summary(first) == "会话一摘要"
    assert memory.get_summary() == "会话二摘要"


def test_delete_session_removes_facts(tmp_path):
    memory = make_memory(tmp_path)
    sid = memory.session_id
    memory.patch_facts({"count": 2})
    memory.delete_session(sid)
    assert memory.get_facts(sid) == {}
