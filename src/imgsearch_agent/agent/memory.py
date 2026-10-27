"""对话记忆与任务状态持久化。

两种模式：
- 纯内存（默认）：开发调试用，进程退出即丢失
- SQLite（可选）：保存每轮对话、trace_id、动作与工具返回，便于回溯失败案例

记忆只负责"存"和"取"，不做决策。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("imgsearch_agent.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    trace_id TEXT,
    action TEXT,
    tool_name TEXT,
    tool_args TEXT,
    tool_result TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Memory:
    def __init__(
        self,
        session_id: str | None = None,
        persist_path: str | Path | None = None,
        resume: bool = False,
    ):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._messages: list[dict[str, str]] = []
        self._db: sqlite3.Connection | None = None
        if persist_path:
            self._db = self._connect(Path(persist_path))
            if resume and session_id is None:
                latest = self._latest_session()
                if latest:
                    self.session_id = latest

    # ── 初始化 ────────────────────────────────────────────────────────────

    def _connect(self, path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.executescript(_SCHEMA)
        conn.commit()
        logger.info("记忆库已连接: %s (session=%s)", path, self.session_id)
        return conn

    # ── 写 ────────────────────────────────────────────────────────────────

    def _latest_session(self) -> str | None:
        """最近一次有消息的会话，用于启动时恢复。"""
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT session_id FROM messages ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def add(self, role: str, content: str) -> None:
        """追加一条对话消息。"""
        item = {"role": role, "content": content, "created_at": _now()}
        self._messages.append(item)
        if self._db is not None:
            self._db.execute(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (self.session_id, role, content, item["created_at"]),
            )
            self._db.commit()

    def add_trace(
        self,
        trace_id: str,
        action: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        tool_result: Any = None,
    ) -> None:
        """记录一次动作与工具返回，便于回溯 bad case。"""
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT INTO traces(session_id, trace_id, action, tool_name, "
                "tool_args, tool_result, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    trace_id,
                    action,
                    tool_name,
                    json.dumps(tool_args, ensure_ascii=False) if tool_args else None,
                    json.dumps(tool_result, ensure_ascii=False)[:4000]
                    if tool_result is not None
                    else None,
                    _now(),
                ),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("写入 trace 失败: %s", e)

    # ── 读 ────────────────────────────────────────────────────────────────

    def recent(self, n: int = 6) -> list[dict[str, str]]:
        """返回最近 n 条对话（role/content），供 Planner 使用。"""
        if self._messages:
            return [
                {"role": m["role"], "content": m["content"]}
                for m in self._messages[-n:]
            ]
        if self._db is not None:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (self.session_id, n),
            ).fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        return []

    def history(self) -> list[dict[str, str]]:
        return self.recent(n=10_000)

    def traces(self, limit: int = 50) -> list[dict[str, Any]]:
        """导出本会话的动作轨迹。"""
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT trace_id, action, tool_name, tool_args, tool_result, created_at "
            "FROM traces WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (self.session_id, limit),
        ).fetchall()
        return [
            {
                "trace_id": r[0],
                "action": r[1],
                "tool_name": r[2],
                "tool_args": json.loads(r[3]) if r[3] else None,
                "tool_result": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    # ── 生命周期 ──────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            finally:
                self._db = None

    def __len__(self) -> int:
        return len(self._messages)
