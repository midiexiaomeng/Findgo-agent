"""停止能力与气泡样式测试。"""

from __future__ import annotations

import inspect

import desktop_app
from imgsearch_agent.retriever.llm.reranker import LLMReranker
from imgsearch_agent.tools import ImageRetrievalTool


def test_tool_accepts_should_stop():
    params = inspect.signature(ImageRetrievalTool.run).parameters
    assert "should_stop" in params


def test_reranker_accepts_should_stop():
    params = inspect.signature(LLMReranker.rerank).parameters
    assert "should_stop" in params


def test_bubble_has_larger_padding():
    win = desktop_app.MainWindow.__new__(desktop_app.MainWindow)
    win._galleries = {}

    user_html = win._chat_html("你", "你好")
    agent_html = win._chat_html("Agent", "找到了")

    for html in (user_html, agent_html):
        assert "padding:14px 18px" in html
        assert "min-width:120px" in html
        assert "border-radius:16px" in html


def test_path_highlight():
    win = desktop_app.MainWindow.__new__(desktop_app.MainWindow)
    win._galleries = {}
    html = win._chat_html("你", r"找 D:\Photos\cat.jpg")
    assert "background:#EAF3EE" in html
    assert "D:\\Photos\\cat.jpg" in html
