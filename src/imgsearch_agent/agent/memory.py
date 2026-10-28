"""对话记忆与任务状态持久化。

存储三类数据：
- sessions：会话元信息（可命名、记录创建与更新时间）
- messages：自然语言对话内容
- tasks：Agent 规划出的任务链（结构化，可回溯与二次编辑）
- traces：工具调用轨迹

产物在 data/agent_sessions.sqlite，按 session_id 隔离。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.schemas import TaskSpec

logger = logging.getLogger("imgsearch_agent.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    images TEXT,
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
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    task_id INTEGER,
    description TEXT,
    query TEXT,
    top_k INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
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
            self._ensure_session_record()

    # ── 初始化 ────────────────────────────────────────────────────────────

    def _connect(self, path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.executescript(_SCHEMA)
        conn.commit()
        self._migrate(conn)
        self._backfill_sessions(conn)
        logger.info("记忆库已连接: %s", path)
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """旧版本 messages 表没有 images 字段，这里补上。"""
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
            if "images" not in cols:
                conn.execute("ALTER TABLE messages ADD COLUMN images TEXT")
                conn.commit()
        except Exception:
            pass

    @staticmethod
    def _backfill_sessions(conn: sqlite3.Connection) -> None:
        """把旧版本只存在 messages 里的会话补进 sessions 表。"""
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id, title, created_at, updated_at) "
                "SELECT session_id, '', MIN(created_at), MAX(created_at) "
                "FROM messages GROUP BY session_id"
            )
            conn.commit()
        except Exception:
            pass

    def _ensure_session_record(self) -> None:
        if self._db is None:
            return
        now = _now()
        try:
            self._db.execute(
                "INSERT OR IGNORE INTO sessions(session_id, title, created_at, updated_at) "
                "VALUES (?, '', ?, ?)",
                (self.session_id, now, now),
            )
            self._db.commit()
        except Exception:
            pass

    def _latest_session(self) -> str | None:
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT session_id FROM messages ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    # ── 会话管理 ──────────────────────────────────────────────────────────

    def new_session(self) -> str:
        self.session_id = uuid.uuid4().hex[:12]
        self._messages = []
        self._ensure_session_record()
        return self.session_id

    def switch_session(self, session_id: str) -> None:
        self.session_id = session_id
        self._messages = []
        self._ensure_session_record()

    def rename_session(self, title: str, session_id: str | None = None) -> None:
        if self._db is None:
            return
        sid = session_id or self.session_id
        try:
            self._db.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title.strip(), _now(), sid),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("重命名会话失败: %s", e)

    def get_title(self, session_id: str | None = None) -> str:
        if self._db is None:
            return ""
        sid = session_id or self.session_id
        try:
            row = self._db.execute(
                "SELECT title FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
            return str(row[0] or "") if row else ""
        except Exception:
            return ""

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        try:
            rows = self._db.execute(
                "SELECT s.session_id, COALESCE(s.title, ''), s.updated_at, "
                "  (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id), "
                "  (SELECT content FROM messages m WHERE m.session_id = s.session_id "
                "     AND m.role = 'user' ORDER BY m.id LIMIT 1) "
                "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception:
            return []

        out: list[dict[str, Any]] = []
        for session_id, title, updated_at, count, first_msg in rows:
            if not title:
                title = (first_msg or "").strip().replace("\n", " ")[:30] or "(空会话)"
            out.append(
                {
                    "session_id": session_id,
                    "title": title,
                    "count": int(count),
                    "updated_at": updated_at,
                }
            )
        return out

    def load_messages(self, session_id: str, limit: int = 300) -> list[dict[str, str]]:
        if self._db is None:
            return []
        try:
            rows = self._db.execute(
                "SELECT role, content, images FROM messages WHERE session_id = ? "
                "ORDER BY id LIMIT ?",
                (session_id, limit),
            ).fetchall()
            out = []
            for r in rows:
                try:
                    images = json.loads(r[2]) if r[2] else []
                except Exception:
                    images = []
                out.append({"role": r[0], "content": r[1], "images": images})
            return out
        except Exception:
            return []

    def delete_session(self, session_id: str) -> None:
        """彻底删除一个会话及其消息、任务、轨迹。"""
        if self._db is None:
            return
        try:
            for table in ("messages", "traces", "tasks", "sessions"):
                self._db.execute(
                    f"DELETE FROM {table} WHERE session_id = ?", (session_id,)
                )
            self._db.commit()
        except Exception as e:
            logger.warning("删除会话失败: %s", e)

    def prune_sessions(self, keep: int) -> int:
        if self._db is None or keep < 0:
            return 0
        sessions = self.list_sessions(limit=1000)
        stale = [s["session_id"] for s in sessions[keep:]]
        if not stale:
            return 0
        try:
            for sid in stale:
                self._db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                self._db.execute("DELETE FROM traces WHERE session_id = ?", (sid,))
                self._db.execute("DELETE FROM tasks WHERE session_id = ?", (sid,))
                self._db.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            self._db.commit()
        except Exception:
            return 0
        return len(stale)

    # ── 任务链持久化 ──────────────────────────────────────────────────────

    def save_tasks(self, tasks: list[TaskSpec], session_id: str | None = None) -> None:
        if self._db is None:
            return
        sid = session_id or self.session_id
        try:
            self._db.execute("DELETE FROM tasks WHERE session_id = ?", (sid,))
            now = _now()
            for t in tasks:
                self._db.execute(
                    "INSERT INTO tasks(session_id, task_id, description, query, top_k, status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (sid, t.id, t.description, t.query, t.top_k, t.status, now),
                )
            self._db.commit()
        except Exception as e:
            logger.warning("保存任务链失败: %s", e)

    def load_tasks(self, session_id: str | None = None) -> list[TaskSpec]:
        if self._db is None:
            return []
        sid = session_id or self.session_id
        try:
            rows = self._db.execute(
                "SELECT task_id, description, query, top_k, status FROM tasks "
                "WHERE session_id = ? ORDER BY id",
                (sid,),
            ).fetchall()
        except Exception:
            return []
        return [
            TaskSpec(
                id=int(r[0] or 0),
                description=str(r[1] or ""),
                query=str(r[2] or ""),
                top_k=int(r[3] or 0),
                status=str(r[4] or "pending"),
            )
            for r in rows
        ]

    # ── 写 ────────────────────────────────────────────────────────────────

    def add(
        self,
        role: str,
        content: str,
        images: list[str] | None = None,
    ) -> None:
        item = {"role": role, "content": content, "created_at": _now()}
        self._messages.append(item)
        if self._db is not None:
            try:
                images_json = (
                    json.dumps(images, ensure_ascii=False) if images else None
                )
                self._db.execute(
                    "INSERT INTO messages(session_id, role, content, images, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (self.session_id, role, content, images_json, item["created_at"]),
                )
                self._db.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (item["created_at"], self.session_id),
                )
                self._db.commit()
            except Exception as e:
                logger.warning("写入消息失败: %s", e)

    def list_traces(
        self, tool_name: str | None = None, limit: int = 500
    ) -> list[dict]:
        """按工具名读取历史轨迹，供留存推荐打分。"""
        if self._db is None:
            return []
        try:
            if tool_name:
                rows = self._db.execute(
                    "SELECT * FROM traces WHERE tool_name = ? ORDER BY id DESC LIMIT ?",
                    (tool_name, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []

    def add_trace(
        self,
        trace_id: str,
        action: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        tool_result: Any = None,
    ) -> None:
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

    def recent(self, n: int = 20) -> list[dict[str, str]]:
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
