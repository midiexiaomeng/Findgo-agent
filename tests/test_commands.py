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
    with pytest.raises(CommandError, match="还没有内容"):
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
