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
    #: 重排模型给出的简短理由，用于向用户解释为什么选它
    reason: str = ""


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


class TaskSpec(BaseModel):
    """一条可分步执行的任务。"""

    id: int = 0
    description: str = ""
    query: str = ""
    top_k: int = Field(0, description="用户明确要求的数量，0 表示用默认值")
    path: str = Field("", description="限定检索的文件夹路径，空表示全库")
    action: str = Field("retrieve", description="步骤类型：scan 扫描目录 / retrieve 检索图片")
    all_results: bool = Field(False, description="是否返回全部匹配结果")
    risk: str = Field("normal", description="风险等级：normal / high")
    reuse_last: bool = Field(False, description="直接复用上一轮的图片结果")
    is_dynamic: bool = Field(False, description="是否由动态工具执行")
    dynamic_spec: dict | None = Field(None, description="动态工具定义，需用户审核后才加载")
    dynamic_args: dict = Field(default_factory=dict, description="动态工具的实际入参")
    # 结构化约束：must / must_not / should，优先于 query 字符串生效
    constraints: dict = Field(
        default_factory=dict,
        description="core_query/must/must_not/should/diversity/retrieve_count",
    )
    # 检索的数据源：library 全库召回，last_results 在上一轮结果里再筛
    input_source: str = Field(
        "library",
        description="library | last_results",
    )
    status: str = Field("pending", description="pending | done | skipped")
