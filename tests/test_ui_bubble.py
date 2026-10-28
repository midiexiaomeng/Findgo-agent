"""界面渲染测试（需要 PyQt5，缺失时跳过）。"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")

import desktop_app  # noqa: E402


def test_bubble_has_larger_padding():
    win = desktop_app.MainWindow.__new__(desktop_app.MainWindow)
    win._galleries = {}

    user_html = win._chat_html("你", "你好")
    agent_html = win._chat_html("Agent", "找到了")

    for html in (user_html, agent_html):
        assert "padding:18px 24px" in html
        assert "width:640px" in html
        assert "border-radius:16px" in html


def test_path_highlight():
    win = desktop_app.MainWindow.__new__(desktop_app.MainWindow)
    win._galleries = {}
    html = win._chat_html("你", r"找 D:\Photos\cat.jpg")
    assert "background:#EAF3EE" in html
    assert "D:\\Photos\\cat.jpg" in html
