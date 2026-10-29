"""斜杠命令执行测试。"""

from __future__ import annotations

import pytest
from findgo.agent.commands import (
    CommandError,
    dispatch,
    is_command,
    run_compact,
    run_diary,
    run_loop,
    run_skill_command,
    split_command,
)


class _FakeMemory:
    def __init__(self, history=None):
        self._history = history or []
        self.summary = ""
        self.added = []

    def history(self):
        return self._history

    def add(self, role, content, images=None):
        self.added.append((role, content))

    def set_summary(self, summary, session_id=None):
        self.summary = summary


class _FakeClient:
    def __init__(self, reply="这是模型生成的正文。"):
        self.reply = reply
        self.calls = []

    def chat(self, messages, json_mode=False, max_tokens=512):
        self.calls.append(messages)
        return self.reply


class _FakeAgent:
    def __init__(self, history=None, reply="这是模型生成的正文。"):
        self.memory = _FakeMemory(history)
        self.client = _FakeClient(reply)
        self.planned = []

    def plan_tasks(self, text):
        from findgo.utils.schemas import TaskSpec

        self.planned.append(text)
        return [TaskSpec(id=1, query=text)]

    def run_tasks(self, tasks, write_memory=True):
        return []


# ── 解析 ──────────────────────────────────────────────────────────────────


def test_split_command_with_args():
    assert split_command("/diary 今天心情不错") == ("/diary", "今天心情不错")


def test_split_command_without_args():
    assert split_command("/compact") == ("/compact", "")


def test_split_command_not_a_command():
    assert split_command("找几张猫的图片") == ("", "找几张猫的图片")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/diary", True),
        ("/diary 补充", True),
        ("/loop 3 找猫", True),
        ("/unknown", False),
        ("找猫", False),
        ("", False),
    ],
)
def test_is_command(text, expected):
    assert is_command(text) is expected


# ── /diary ────────────────────────────────────────────────────────────────


def test_diary_requires_history(tmp_path, monkeypatch):
    agent = _FakeAgent(history=[])
    monkeypatch.setattr("findgo.agent.commands._data_dir", lambda _a: tmp_path)
    with pytest.raises(CommandError, match="还没有聊过"):
        run_diary(agent, "")


def test_diary_writes_file(tmp_path, monkeypatch):
    history = [
        {"role": "user", "content": "帮我找猫的图片"},
        {"role": "assistant", "content": "找到 5 张"},
    ]
    agent = _FakeAgent(history=history, reply="今天下午花了一小时整理图片库。")
    monkeypatch.setattr("findgo.agent.commands._data_dir", lambda _a: tmp_path)
    monkeypatch.setattr("findgo.agent.commands._project_root", lambda _a: tmp_path)

    result = run_diary(agent, "顺便记一句天气不错")

    files = list((tmp_path / "diary").glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "今天下午花了一小时整理图片库。" in content
    assert "日记" in result


def test_messages_since_reads_across_sessions(tmp_path):
    from findgo.agent.memory import Memory

    db = tmp_path / "s.db"
    first = Memory(persist_path=db)
    first.add("user", "早上找猫")
    first.add("assistant", "找到 3 张")
    first.close()

    second = Memory(persist_path=db)
    second.add("user", "下午找狗")
    second.add("assistant", "找到 2 张")

    rows = second.messages_since("2000-01-01T00:00:00")
    contents = [r["content"] for r in rows]

    assert "早上找猫" in contents
    assert "下午找狗" in contents
    second.close()


def test_messages_since_filters_by_time(tmp_path):
    from findgo.agent.memory import Memory

    m = Memory(persist_path=tmp_path / "s.db")
    m.add("user", "很久以前")

    # 未来时间点应该读不到
    assert m.messages_since("2099-01-01T00:00:00") == []
    m.close()


def test_diary_uses_today_across_sessions(tmp_path, monkeypatch):
    """日记要读今天所有会话，不是只看当前会话。"""
    from findgo.agent.memory import Memory

    db = tmp_path / "s.db"
    first = Memory(persist_path=db)
    first.add("user", "上午找猫")
    first.add("assistant", "找到 3 张")
    first.close()

    second = Memory(persist_path=db)
    second.add("user", "下午找狗")
    second.add("assistant", "找到 2 张")

    monkeypatch.setattr("findgo.agent.commands._data_dir", lambda _a: tmp_path)
    monkeypatch.setattr("findgo.agent.commands._project_root", lambda _a: tmp_path)

    agent = _FakeAgent(history=[])
    agent.memory = second
    agent.client = _FakeClient("今天很忙。")

    run_diary(agent, "")

    prompt = agent.client.calls[0][1]["content"]
    # 两个会话的内容都要在里面
    assert "上午找猫" in prompt
    assert "下午找狗" in prompt
    second.close()


def test_diary_prompt_is_assistant_voice(tmp_path, monkeypatch):
    """日记必须是助手口吻，不是用户自己的日记。"""
    history = [
        {"role": "user", "content": "找 5 张猫的图片"},
        {"role": "assistant", "content": "找到 5 张"},
    ]
    agent = _FakeAgent(history=history)
    monkeypatch.setattr("findgo.agent.commands._data_dir", lambda _a: tmp_path)
    monkeypatch.setattr("findgo.agent.commands._project_root", lambda _a: tmp_path)

    run_diary(agent, "")

    prompt = agent.client.calls[0][1]["content"]
    assert "我（助手）的第一人称视角" in prompt
    assert "你是助手，不是主人" in prompt
    assert "主人" in prompt


def test_diary_prompt_forbids_fabrication(tmp_path, monkeypatch):
    """日记提示词必须明确禁止编造记录里没有的内容。"""
    history = [
        {"role": "user", "content": "找 5 张猫的图片"},
        {"role": "assistant", "content": "找到 5 张"},
    ]
    agent = _FakeAgent(history=history)
    monkeypatch.setattr("findgo.agent.commands._data_dir", lambda _a: tmp_path)
    monkeypatch.setattr("findgo.agent.commands._project_root", lambda _a: tmp_path)

    run_diary(agent, "")

    prompt = agent.client.calls[0][1]["content"]
    assert "不能写" in prompt
    assert "不允许虚构" in prompt
    assert "找 5 张猫的图片" in prompt


def test_diary_appends_same_day(tmp_path, monkeypatch):
    history = [{"role": "user", "content": "找图"}]
    agent = _FakeAgent(history=history, reply="第一次")
    monkeypatch.setattr("findgo.agent.commands._data_dir", lambda _a: tmp_path)
    monkeypatch.setattr("findgo.agent.commands._project_root", lambda _a: tmp_path)

    run_diary(agent, "")
    agent.client.reply = "第二次"
    run_diary(agent, "")

    files = list((tmp_path / "diary").glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "第一次" in content and "第二次" in content


# ── /compact ──────────────────────────────────────────────────────────────


def test_compact_needs_enough_messages():
    agent = _FakeAgent(history=[{"role": "user", "content": "hi"}])
    with pytest.raises(CommandError, match="太短"):
        run_compact(agent, "")


def test_compact_writes_summary():
    history = [
        {"role": "user", "content": "找猫的图片"},
        {"role": "assistant", "content": "找到 3 张"},
        {"role": "user", "content": "只要横图"},
        {"role": "assistant", "content": "已筛选"},
    ]
    agent = _FakeAgent(history=history, reply="用户要横版的猫图，已筛出 3 张。")
    result = run_compact(agent, "")
    assert agent.memory.summary == "用户要横版的猫图，已筛出 3 张。"
    assert "压缩" in result


# ── /loop ─────────────────────────────────────────────────────────────────


def test_loop_needs_args():
    agent = _FakeAgent()
    with pytest.raises(CommandError, match="用法"):
        run_loop(agent, "")


def test_loop_parses_count():
    agent = _FakeAgent()
    result = run_loop(agent, "2 找猫")
    assert agent.planned == ["找猫"]
    assert "2 次" in result


def test_loop_clamps_count():
    agent = _FakeAgent()
    run_loop(agent, "99 找猫")
    assert agent.planned == ["找猫"]


# ── 分派 ──────────────────────────────────────────────────────────────────


def test_dispatch_unknown_returns_none():
    assert dispatch("/nothing", _FakeAgent()) is None


def test_dispatch_plain_text_returns_none():
    assert dispatch("找猫", _FakeAgent()) is None


def test_dispatch_skill_command():
    reply = dispatch("/office-documents", _FakeAgent())
    assert "需要一段需求描述" in reply


def test_dispatch_skill_with_args():
    reply = dispatch("/hana-plugin-creator 做个插件", _FakeAgent())
    assert "做个插件" in reply


# ── 权限菜单 ──────────────────────────────────────────────────────────────


def test_permission_menu_lists_four():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from findgo.ui.command_panel import PermissionMenu

    menu = PermissionMenu("ask")
    assert len(menu.actions()) == 4
    texts = [a.text() for a in menu.actions()]
    assert any("完整权限" in t for t in texts)
    assert any("只读模式" in t for t in texts)
    assert any(t.startswith("●") for t in texts)
