"""停止能力测试（不依赖 PyQt5）。"""

from __future__ import annotations

import inspect

from findgo.retriever.llm.reranker import LLMReranker
from findgo.tools import ImageRetrievalTool


def test_tool_accepts_should_stop():
    params = inspect.signature(ImageRetrievalTool.run).parameters
    assert "should_stop" in params


def test_reranker_accepts_should_stop():
    params = inspect.signature(LLMReranker.rerank).parameters
    assert "should_stop" in params
