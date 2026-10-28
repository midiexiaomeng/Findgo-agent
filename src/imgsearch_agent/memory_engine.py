"""三层记忆引擎：短期会话 / 摘要压缩 / 长期向量记忆。

设计目标：和 Agent 主逻辑解耦，通过 ``MemoryEngine`` 一个门面接入。

    短期记忆   按 session_id 隔离的消息列表，sqlite 落地
    摘要记忆   消息超过阈值时用 LLM 压缩，原文保留备份
    长期记忆   只把有价值的事实写入 Qdrant，按 user_id 隔离检索

用法：
    engine = MemoryEngine()
    engine.add_message(user_id, session_id, "user", "找两张猫")
    messages = engine.build_messages(user_id, session_id, system_prompt, query)
    engine.after_turn(user_id, session_id, user_text, assistant_text)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("imgsearch_agent.memory_engine")


# ────────────────────────────────────────────────────────────────────────────
# 配置：集中放头部，接入方按需覆盖
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryConfig:
    # 存储
    db_path: str = "./data/memory.db"
    qdrant_path: str = "./data/qdrant"
    collection: str = "memories"

    # 短期记忆
    max_turns: int = 20            # 超过这么多轮触发摘要压缩
    recent_turns: int = 6          # 组装 prompt 时保留的最近轮数

    # 长期记忆
    top_k: int = 5                 # 每次检索注入多少条记忆
    similarity_threshold: float = 0.25
    memory_enabled: bool = True    # 全局开关：关闭后不写也不读长期记忆

    # 提示词模板（可配置，不硬编码在逻辑里）
    memory_prefix: str = (
        "【历史记忆参考，如和当前用户最新指令冲突，以用户最新指令优先】\n"
        "{memories}"
    )
    summary_prefix: str = "【前情摘要】\n{summary}"

    summary_prompt: str = (
        "请把下面的对话压缩成事实摘要，只保留这些内容：\n"
        "1. 用户的核心需求和约束偏好\n"
        "2. 已经完成的任务和结论\n"
        "3. 工具返回的关键结果\n"
        "4. 失败原因和踩过的坑\n"
        "不要保留寒暄和重复表述，控制在 300 字以内。\n\n"
        "对话：\n{conversation}"
    )

    extract_prompt: str = (
        "判断下面这轮对话里有没有值得长期记住的事实。\n"
        "值得记住的只有：用户固定偏好、历史任务结论、参数配置、踩过的坑、长期约束。\n"
        "闲聊、一次性问答、临时查询都不要。\n\n"
        "用户：{user}\n助手：{assistant}\n\n"
        '只输出 JSON：{{"useful": true/false, "facts": ["事实1", "事实2"], '
        '"tags": ["preference" | "task" | "error"]}}'
    )


# ────────────────────────────────────────────────────────────────────────────
# 短期记忆：sqlite
# ────────────────────────────────────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class SessionStore:
    """短期会话记忆，按 session_id 隔离。"""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def append(self, user_id: str, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(user_id, session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, session_id, role, content, time.time()),
            )

    def recent(self, session_id: str, limit: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def all_messages(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def count(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["n"])

    def get_summary(self, session_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
        return str(row["summary"]) if row else ""

    def set_summary(self, session_id: str, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summaries(session_id, summary, updated_at) "
                "VALUES (?, ?, ?)",
                (session_id, summary, time.time()),
            )


# ────────────────────────────────────────────────────────────────────────────
# 长期记忆：Qdrant
# ────────────────────────────────────────────────────────────────────────────


class LongTermMemory:
    """长期向量记忆，只存抽取出来的事实。"""

    def __init__(self, config: MemoryConfig, store=None):
        self.config = config
        self._store = store

    @property
    def store(self):
        if self._store is None:
            from .retriever.config import Config
            from .retriever.vectorstore.qdrant_store import QdrantVectorStore

            cfg = Config.from_yaml()
            self._store = QdrantVectorStore(
                collection=self.config.collection,
                embedding_dim=int(cfg.embedding.embedding_dim),
                mode=cfg.vector_store.qdrant.mode,
                path=cfg.resolve(cfg.vector_store.qdrant.path),
            )
        return self._store

    def add(
        self,
        user_id: str,
        session_id: str,
        text: str,
        vector,
        tags: list[str] | None = None,
    ) -> str:
        memory_id = f"mem-{uuid.uuid4().hex[:16]}"
        self.store.add(
            [memory_id],
            vector.reshape(1, -1),
            [
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "text": text,
                    "tags": ",".join(tags or []),
                    "created_at": time.time(),
                }
            ],
        )
        logger.info("长期记忆写入 user=%s session=%s", user_id, session_id)
        return memory_id

    def search(self, user_id: str, vector, top_k: int, threshold: float) -> list[dict]:
        hits = self.store.query(vector, top_k=max(top_k * 3, 10))
        results = []
        for hit in hits:
            meta = hit.metadata
            if str(meta.get("user_id", "")) != user_id:
                continue
            if float(hit.score) < threshold:
                continue
            results.append(
                {
                    "id": hit.id,
                    "text": meta.get("text", ""),
                    "tags": meta.get("tags", ""),
                    "score": round(float(hit.score), 4),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def delete(self, memory_id: str) -> None:
        self.store.delete([memory_id])
        logger.info("长期记忆删除 %s", memory_id)

    def clear(self, user_id: str) -> int:
        all_ids = self.store.get_all_ids()
        if not all_ids:
            return 0

        records = self.store.get(ids=all_ids)
        targets = [
            r.id for r in records if str(r.metadata.get("user_id", "")) == user_id
        ]
        if targets:
            self.store.delete(targets)
        logger.info("清除用户 %s 的长期记忆 %d 条", user_id, len(targets))
        return len(targets)

    def list_for_user(self, user_id: str) -> list[dict]:
        all_ids = self.store.get_all_ids()
        if not all_ids:
            return []
        records = self.store.get(ids=all_ids)
        return [
            {
                "id": r.id,
                "text": r.metadata.get("text", ""),
                "tags": r.metadata.get("tags", ""),
                "session_id": r.metadata.get("session_id", ""),
                "created_at": r.metadata.get("created_at", 0),
            }
            for r in records
            if str(r.metadata.get("user_id", "")) == user_id
        ]


# ────────────────────────────────────────────────────────────────────────────
# 门面
# ────────────────────────────────────────────────────────────────────────────


class MemoryEngine:
    """三层记忆的统一入口。"""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        encoder=None,
        llm=None,
        store=None,
    ):
        self.config = config or MemoryConfig()
        self.sessions = SessionStore(self.config.db_path)
        self.long_term = LongTermMemory(self.config, store=store)
        self._encoder = encoder
        self._llm = llm

    # ── 依赖 ────────────────────────────────────────────────────────────────

    @property
    def encoder(self):
        if self._encoder is None:
            from .retriever import build_pipeline

            self._encoder = build_pipeline().encoder
        return self._encoder

    @property
    def llm(self):
        if self._llm is None:
            from .config import AgentConfig
            from .llm import ChatClient

            self._llm = ChatClient(AgentConfig().llm)
        return self._llm

    # ── 短期记忆 ────────────────────────────────────────────────────────────

    def add_message(
        self, user_id: str, session_id: str, role: str, content: str
    ) -> None:
        self.sessions.append(user_id, session_id, role, content)

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.all_messages(session_id)

    # ── 长期记忆检索 ────────────────────────────────────────────────────────

    def search(self, user_id: str, query: str) -> list[dict]:
        if not self.config.memory_enabled:
            return []
        vector = self.encoder.encode_text(query)
        hits = self.long_term.search(
            user_id,
            vector,
            self.config.top_k,
            self.config.similarity_threshold,
        )
        if hits:
            logger.info("命中长期记忆 %d 条", len(hits))
        return hits

    # ── 组装 prompt ─────────────────────────────────────────────────────────

    def build_messages(
        self,
        user_id: str,
        session_id: str,
        system_prompt: str,
        query: str,
    ) -> list[dict]:
        blocks: list[str] = [system_prompt]

        if self.config.memory_enabled:
            memories = self.search(user_id, query)
            if memories:
                text = "\n".join(f"- {m['text']}" for m in memories)
                blocks.append(self.config.memory_prefix.format(memories=text))

        summary = self.sessions.get_summary(session_id)
        if summary:
            blocks.append(self.config.summary_prefix.format(summary=summary))

        messages = [{"role": "system", "content": "\n\n".join(blocks)}]
        recent = self.sessions.recent(
            session_id, max(1, self.config.recent_turns) * 2
        )
        messages.extend(recent)
        messages.append({"role": "user", "content": query})
        return messages

    # ── 摘要压缩 ────────────────────────────────────────────────────────────

    def summarize(self, session_id: str) -> str:
        history = self.sessions.all_messages(session_id)
        if not history:
            return ""

        conversation = "\n".join(f"{m['role']}：{m['content']}" for m in history)
        prompt = self.config.summary_prompt.format(conversation=conversation[:8000])

        try:
            summary = self.llm.chat(
                [{"role": "user", "content": prompt}], max_tokens=512
            ).strip()
        except Exception as e:
            logger.warning("摘要生成失败: %s", e)
            return ""

        if summary:
            self.sessions.set_summary(session_id, summary)
            logger.info("会话 %s 摘要已更新（%d 字）", session_id, len(summary))
        return summary

    # ── 事实抽取 ────────────────────────────────────────────────────────────

    def extract_facts(self, user_text: str, assistant_text: str) -> dict:
        prompt = self.config.extract_prompt.format(
            user=user_text[:2000], assistant=assistant_text[:2000]
        )
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                max_tokens=400,
            )
            data = json.loads(raw.strip())
        except Exception as e:
            logger.debug("事实抽取失败: %s", e)
            return {"useful": False, "facts": [], "tags": []}

        if not isinstance(data, dict):
            return {"useful": False, "facts": [], "tags": []}
        return {
            "useful": bool(data.get("useful")),
            "facts": [str(f) for f in (data.get("facts") or []) if f],
            "tags": [str(t) for t in (data.get("tags") or []) if t],
        }

    # ── 一轮结束后的动作 ────────────────────────────────────────────────────

    def after_turn(
        self, user_id: str, session_id: str, user_text: str, assistant_text: str
    ) -> dict:
        result = {"facts_saved": 0, "summarized": False}

        if self.config.memory_enabled:
            extracted = self.extract_facts(user_text, assistant_text)
            if extracted["useful"] and extracted["facts"]:
                for fact in extracted["facts"]:
                    vector = self.encoder.encode_text(fact)
                    self.long_term.add(
                        user_id,
                        session_id,
                        fact,
                        vector,
                        extracted["tags"],
                    )
                result["facts_saved"] = len(extracted["facts"])
                logger.info("本轮抽取 %d 条长期记忆", result["facts_saved"])

        if self.sessions.count(session_id) >= self.config.max_turns * 2:
            if self.summarize(session_id):
                result["summarized"] = True

        return result

    # ── 管理接口 ────────────────────────────────────────────────────────────

    def list_memories(self, user_id: str) -> list[dict]:
        return self.long_term.list_for_user(user_id)

    def delete_memory(self, memory_id: str) -> None:
        self.long_term.delete(memory_id)

    def clear_memories(self, user_id: str) -> int:
        return self.long_term.clear(user_id)

    def set_enabled(self, enabled: bool) -> None:
        self.config.memory_enabled = bool(enabled)
        logger.info("长期记忆开关：%s", "开启" if enabled else "关闭")
