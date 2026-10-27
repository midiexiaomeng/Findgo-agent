"""工具抽象基类。

所有工具继承 BaseTool，规定 name、description、args_schema、run() 四个契约。
Agent 只依赖这个接口，新增工具（OCR、图片下载等）无需改动 Agent 主流程。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] | None = None

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """执行工具，返回结构化结果。"""

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """用 args_schema 校验入参；非法参数抛 ValidationError，不进入 run()。"""
        from ..utils.exceptions import ValidationError

        if self.args_schema is None:
            return dict(args)

        try:
            model = self.args_schema(**args)
        except Exception as e:
            raise ValidationError(f"工具 {self.name} 参数不合法: {e}") from e
        return model.model_dump()

    def signature(self) -> str:
        return f"{self.name} — {self.description}"
