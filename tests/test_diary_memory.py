"""日记读取与记忆注入测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

from findgo.agent.diary import diary_dir, recent_diary_text


def _write_diary(root, day_offset: int, content: str) -> None:
    folder = root / "data" / "diary"
    folder.mkdir(parents=True, exist_ok=True)
    day = datetime.now().date() - timedelta(days=day_offset)
    (folder / f"{day.isoformat()}.md").write_text(content, encoding="utf-8")


def test_diary_dir_default():
    assert diary_dir().name == "diary"


def test_recent_diary_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("findgo.agent.diary.diary_dir", lambda root=None: tmp_path / "none")
    assert recent_diary_text(days=3) == ""


def test_recent_diary_reads_today(tmp_path, monkeypatch):
    _write_diary(tmp_path, 0, "# 今天\n\n找了五张猫的图片。")
    monkeypatch.setattr(
        "findgo.agent.diary.diary_dir", lambda root=None: tmp_path / "data" / "diary"
    )

    text = recent_diary_text(days=3)
    assert "找了五张猫的图片" in text


def test_recent_diary_orders_old_to_new(tmp_path, monkeypatch):
    _write_diary(tmp_path, 2, "前天的内容")
    _write_diary(tmp_path, 0, "今天的内容")
    monkeypatch.setattr(
        "findgo.agent.diary.diary_dir", lambda root=None: tmp_path / "data" / "diary"
    )

    text = recent_diary_text(days=3)
    assert text.index("前天的内容") < text.index("今天的内容")


def test_recent_diary_skips_old(tmp_path, monkeypatch):
    _write_diary(tmp_path, 10, "十天前的内容")
    _write_diary(tmp_path, 0, "今天的内容")
    monkeypatch.setattr(
        "findgo.agent.diary.diary_dir", lambda root=None: tmp_path / "data" / "diary"
    )

    text = recent_diary_text(days=3)
    assert "十天前" not in text
    assert "今天的内容" in text


def test_recent_diary_respects_budget(tmp_path, monkeypatch):
    _write_diary(tmp_path, 1, "A" * 1500)
    _write_diary(tmp_path, 0, "B" * 1500)
    monkeypatch.setattr(
        "findgo.agent.diary.diary_dir", lambda root=None: tmp_path / "data" / "diary"
    )

    text = recent_diary_text(days=3, max_chars=1600)
    # 预算只够装最近一天
    assert "B" * 100 in text
    assert len(text) < 3200


# ── Planner 注入 ──────────────────────────────────────────────────────────


def test_plan_tasks_accepts_extra_context():
    import inspect

    from findgo.agent.planner import Planner

    params = inspect.signature(Planner.plan_tasks).parameters
    assert "extra_context" in params


def test_memory_context_includes_facts(tmp_path):
    from findgo.agent.orchestrator import FindgoAgent
    from findgo.utils.schemas import TaskSpec

    orch = FindgoAgent.__new__(FindgoAgent)

    class _Mem:
        def get_facts(self):
            return {"count": 3, "filters": {"color": "黑"}}

        def recent(self, n=20):
            return []

    orch.memory = _Mem()
    orch.tools = {}

    ctx = orch._memory_context()
    assert "当前会话事实" in ctx
    assert '"count": 3' in ctx


def test_planner_query_instruction_excludes_count():
    """提示词必须要求 query 去掉数量词，否则 CLIP 会理解成图里有几只猫。"""
    import inspect

    src = inspect.getsource(
        __import__("findgo.agent.planner", fromlist=["Planner"]).Planner.plan_tasks
    )
    assert "不要带数量词" in src
    assert "与图片内容里有几个对象无关" in src


def test_memory_context_includes_diary(tmp_path, monkeypatch):
    from findgo.agent.diary import recent_diary_text

    _write_diary(tmp_path, 0, "今天找了几张猫的图片")
    monkeypatch.setattr(
        "findgo.agent.diary.diary_dir", lambda root=None: tmp_path / "data" / "diary"
    )

    from findgo.agent.orchestrator import FindgoAgent

    orch = FindgoAgent.__new__(FindgoAgent)

    class _Mem:
        def get_facts(self):
            return {}

        def recent(self, n=20):
            return []

    orch.memory = _Mem()
    orch.tools = {}

    ctx = orch._memory_context()
    assert "最近几天的日记" in ctx
    assert "今天找了几张猫的图片" in ctx


def test_session_survives_restart(tmp_path):
    """关掉再开，同一个 session 的对话要能读回来。"""
    from findgo.agent.memory import Memory

    db = tmp_path / "sessions.sqlite"

    first = Memory(persist_path=db)
    sid = first.session_id
    first.add("user", "帮我找两张猫的图片")
    first.add("assistant", "找到 2 张")

    second = Memory(persist_path=db, resume=True)
    assert second.session_id == sid

    history = second.history()
    assert len(history) == 2
    assert "两张猫" in history[0]["content"]


def test_session_facts_survive_restart(tmp_path):
    from findgo.agent.memory import Memory

    db = tmp_path / "sessions.sqlite"
    first = Memory(persist_path=db)
    sid = first.session_id
    first.patch_facts({"count": 2})

    second = Memory(persist_path=db, session_id=sid)
    assert second.get_facts() == {"count": 2}


def test_task_spec_top_k_default():
    from findgo.utils.schemas import TaskSpec

    assert TaskSpec().top_k == 0
