"""界面渲染测试（需要 PyQt5，缺失时跳过）。"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")

import desktop_app  # noqa: E402


def test_global_qss_imported():
    """main() 会调 app.setStyleSheet(GLOBAL_QSS)，导入丢失会直接启动失败。"""
    assert hasattr(desktop_app, "GLOBAL_QSS")
    assert isinstance(desktop_app.GLOBAL_QSS, str)
    assert desktop_app.GLOBAL_QSS.strip()


def test_main_window_constructible():
    """MainWindow 能用 __new__ 建实例，不触发重初始化。"""
    win = desktop_app.MainWindow.__new__(desktop_app.MainWindow)
    assert win is not None


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
