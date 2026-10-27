"""带 trace_id 的全链路日志。

每个会话分配唯一 trace_id，日志字段统一为：
    trace_id | module | action | 自定义字段
排查问题只需按 trace_id 过滤。
"""

from __future__ import annotations

import logging
import uuid

_LOGGER = logging.getLogger("imgsearch_agent")


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("imgsearch_agent")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False


class TraceLogger:
    """会话级日志器，持有唯一 trace_id。"""

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:12]

    def _emit(self, level: int, module: str, action: str, **fields) -> None:
        parts = [
            f"trace_id={self.trace_id}",
            f"module={module}",
            f"action={action}",
        ]
        parts += [f"{k}={v}" for k, v in fields.items()]
        _LOGGER.log(level, " | ".join(parts))

    def info(self, module: str, action: str, **fields) -> None:
        self._emit(logging.INFO, module, action, **fields)

    def warning(self, module: str, action: str, **fields) -> None:
        self._emit(logging.WARNING, module, action, **fields)

    def error(self, module: str, action: str, **fields) -> None:
        self._emit(logging.ERROR, module, action, **fields)
