"""统一异常定义，区分不同故障来源。"""


class AgentError(Exception):
    """Agent 基础异常。"""


class LLMError(AgentError):
    """大模型调用失败。"""


class VectorStoreError(AgentError):
    """向量库连接或查询失败。"""


class ToolError(AgentError):
    """工具执行失败。"""


class ValidationError(AgentError):
    """参数校验失败。"""
