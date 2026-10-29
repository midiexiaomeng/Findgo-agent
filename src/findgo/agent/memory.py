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
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.schemas import TaskSpec

logger = logging.getLogger("findgo.memory")

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
CREATE TABLE IF NOT EXISTS facts (
    session_id TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:
        return "unknown"


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """递归合并：值为 None 表示删除该字段，其余覆盖或新增。"""
    for key, value in patch.items():
        if value is None:
            base.pop(key, None)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


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
        # 设为 Row：既支持下标，也能 dict(row)，供 trace 读取用
        conn.row_factory = sqlite3.Row
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
            for table in ("messages", "traces", "tasks", "facts", "summaries", "sessions"):
                self._db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
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
                self._db.execute("DELETE FROM facts WHERE session_id = ?", (sid,))
                self._db.execute("DELETE FROM summaries WHERE session_id = ?", (sid,))
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
                images_json = json.dumps(images, ensure_ascii=False) if images else None
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
        self,
        tool_name: str | None = None,
        limit: int = 500,
        session_id: str | None = None,
    ) -> list[dict]:
        """按工具名读取历史轨迹，供留存推荐打分。

        session_id 为空时按当前会话过滤；传 "*" 则跨会话读全部。
        """
        if self._db is None:
            return []

        scope = session_id if session_id is not None else self.session_id
        all_sessions = scope == "*"

        try:
            if all_sessions and tool_name:
                rows = self._db.execute(
                    "SELECT * FROM traces WHERE tool_name = ? ORDER BY id DESC LIMIT ?",
                    (tool_name, limit),
                ).fetchall()
            elif all_sessions:
                rows = self._db.execute(
                    "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            elif tool_name:
                rows = self._db.execute(
                    "SELECT * FROM traces WHERE session_id = ? AND tool_name = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (scope, tool_name, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM traces WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (scope, limit),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []

    def session_traces(self, limit: int = 200) -> list[dict]:
        """当前会话的执行轨迹，按时间正序，供回放面板使用。"""
        rows = self.list_traces(limit=limit)
        return list(reversed(rows))

    # ── 导出 ────────────────────────────────────────────────────────────

    def export_session(self, path: str | Path | None = None) -> Path:
        """把当前会话导出成一份可复现的 JSON 快照。

        包含消息、任务链、轨迹、结构化事实、摘要。
        密码类信息不在其中，但图路径会包含，提交 issue 前自己看一眼。
        """
        snapshot = {
            "session_id": self.session_id,
            "title": self.get_title(),
            "exported_at": _now(),
            "version": _version(),
            "messages": self.history(),
            "tasks": self.load_tasks(),
            "traces": self.session_traces(limit=1000),
            "facts": self.get_facts(),
            "summary": self.get_summary(),
        }

        if path is None:
            base = Path("data") / "exports"
            base.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = base / f"trace_{self.session_id[:8]}_{stamp}.json"

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("会话已导出: %s", target)
        return target

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
                    (
                        json.dumps(tool_result, ensure_ascii=False)[:4000]
                        if tool_result is not None
                        else None
                    ),
                    _now(),
                ),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("写入 trace 失败: %s", e)

    # ── 结构化事实（本会话的唯一生效约束源）──────────────────────

    def get_facts(self, session_id: str | None = None) -> dict[str, Any]:
        if self._db is None:
            return {}
        sid = session_id or self.session_id
        try:
            row = self._db.execute("SELECT data FROM facts WHERE session_id = ?", (sid,)).fetchone()
            return json.loads(row[0]) if row and row[0] else {}
        except Exception:
            return {}

    def set_facts(self, data: dict[str, Any], session_id: str | None = None) -> None:
        if self._db is None:
            return
        sid = session_id or self.session_id
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO facts(session_id, data, updated_at) " "VALUES (?, ?, ?)",
                (sid, json.dumps(data, ensure_ascii=False), _now()),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("写入结构化事实失败: %s", e)

    def patch_facts(self, patch: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        """按 patch 增量更新事实：值为 null 表示删除该字段。"""
        current = self.get_facts(session_id)
        merged = _deep_merge(current, patch or {})
        self.set_facts(merged, session_id)
        logger.info("结构化事实已更新，字段数=%d", len(merged))
        return merged

    def clear_facts(self, session_id: str | None = None) -> None:
        if self._db is None:
            return
        sid = session_id or self.session_id
        try:
            self._db.execute("DELETE FROM facts WHERE session_id = ?", (sid,))
            self._db.commit()
            logger.info("已清空会话 %s 的结构化事实", sid)
        except Exception as e:
            logger.warning("清空事实失败: %s", e)

    # ── 摘要 ────────────────────────────────────────────────────────────

    def get_summary(self, session_id: str | None = None) -> str:
        if self._db is None:
            return ""
        sid = session_id or self.session_id
        try:
            row = self._db.execute(
                "SELECT summary FROM summaries WHERE session_id = ?", (sid,)
            ).fetchone()
            return str(row[0]) if row else ""
        except Exception:
            return ""

    def set_summary(self, summary: str, session_id: str | None = None) -> None:
        if self._db is None:
            return
        sid = session_id or self.session_id
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO summaries(session_id, summary, updated_at) "
                "VALUES (?, ?, ?)",
                (sid, summary, _now()),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("写入摘要失败: %s", e)

    # ── 读 ────────────────────────────────────────────────────────────────

    def recent(self, n: int = 20) -> list[dict[str, str]]:
        if self._messages:
            return [{"role": m["role"], "content": m["content"]} for m in self._messages[-n:]]
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
