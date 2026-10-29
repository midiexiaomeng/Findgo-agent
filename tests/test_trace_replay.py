"""执行轨迹读取与回放测试。"""

from __future__ import annotations

import pytest
from findgo.agent.memory import Memory


@pytest.fixture()
def mem(tmp_path):
    m = Memory(persist_path=tmp_path / "sessions.sqlite")
    yield m
    m.close()


def test_session_traces_empty(mem):
    assert mem.session_traces() == []


def test_add_and_read_trace(mem):
    mem.add_trace("t1", "tool_call", "image_retrieval", {"query": "猫"}, {"images": 3})
    traces = mem.session_traces()

    assert len(traces) == 1
    # 回放面板需要时间正序
    assert traces[0]["tool_name"] == "image_retrieval"


def test_session_traces_ordered_ascending(mem):
    mem.add_trace("t1", "tool_call", "a", {}, {})
    mem.add_trace("t2", "critic_rewrite", "b", {}, {})
    mem.add_trace("t3", "tool_call", "c", {}, {})

    traces = mem.session_traces()
    assert [t["tool_name"] for t in traces] == ["a", "b", "c"]


def test_traces_scoped_to_session(mem, tmp_path):
    mem.add_trace("t1", "tool_call", "a", {}, {})

    other = Memory(persist_path=tmp_path / "sessions.sqlite", session_id="other-session")
    other.add_trace("t2", "tool_call", "b", {}, {})
    other.close()

    mem.switch_session(mem.session_id)
    assert [t["tool_name"] for t in mem.session_traces()] == ["a"]


def test_traces_across_sessions(mem):
    mem.add_trace("t1", "tool_call", "a", {}, {})
    all_traces = mem.list_traces(session_id="*")
    assert len(all_traces) >= 1


def test_trace_payload_truncated(mem):
    big = {"data": "x" * 10000}
    mem.add_trace("t1", "tool_call", "a", {"q": "x"}, big)
    traces = mem.session_traces()

    assert len(traces[0]["tool_result"]) <= 4100


# ── UI ────────────────────────────────────────────────────────────────────


def test_trace_dialog_constructs():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from findgo.ui.trace_panel import TraceDialog

    traces = [
        {
            "created_at": "2026-10-29 15:00:00",
            "action": "tool_call",
            "tool_name": "image_retrieval",
            "trace_id": "abc123",
            "tool_args": '{"query": "猫"}',
            "tool_result": '{"images": 2}',
        }
    ]
    dialog = TraceDialog(traces)
    assert dialog.timeline.count() == 1
    assert "image_retrieval" in dialog.detail.toPlainText()
    dialog.close()


def test_trace_dialog_empty():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from findgo.ui.trace_panel import TraceDialog

    dialog = TraceDialog([])
    assert dialog.timeline.count() == 1
    assert "还没有执行记录" in dialog.timeline.item(0).text()
    dialog.close()


def test_trace_pretty_handles_bad_json():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from findgo.ui.trace_panel import TraceDialog

    assert TraceDialog._pretty(None) == "（空）"
    assert "not-json" in TraceDialog._pretty("not-json")
