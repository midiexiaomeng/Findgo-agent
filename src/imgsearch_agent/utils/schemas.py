"""结构化数据模型：统一 Agent 的动作、任务状态、工具输入输出。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """任务状态枚举。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    NEED_HUMAN = "need_human"


class AgentAction(BaseModel):
    """Planner 输出的单个动作，强制结构化，拒绝裸字符串解析。"""

    action_type: str = Field(..., description="tool_call | ask_user | answer")
    tool_name: Optional[str] = Field(None, description="action_type=tool_call 时必填")
    tool_args: dict[str, Any] = Field(default_factory=dict)
    thought: str = Field("", description="思考过程，用于日志与调试")
    reply: str = Field("", description="ask_user / answer 时的回复文本")


class RetrievedImage(BaseModel):
    """单张检索结果。"""

    rank: int = 0
    path: str = ""
    caption: str = ""
    score: float = 0.0


class RetrievalResult(BaseModel):
    """工具返回的检索结果。"""

    query: str = ""
    images: list[RetrievedImage] = Field(default_factory=list)
    confidence: float = Field(0.0, description="平均相似度，供 Critic 判定")
    summary: str = ""


class AgentState(BaseModel):
    """一次会话的运行状态，可持久化。"""

    session_id: str
    status: TaskStatus = TaskStatus.PENDING
    query: str = ""
    iterations: int = 0
    retries: int = 0
    final_answer: str = ""
