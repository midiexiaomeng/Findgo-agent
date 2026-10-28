"""环境自检测试。"""

from __future__ import annotations

from findgo.diagnostics import run_checks, summarize


def test_run_checks_returns_items():
    checks = run_checks()
    assert checks
    assert all({"name", "ok", "detail"} <= set(c) for c in checks)


def test_checks_have_bool_status():
    for check in run_checks():
        assert isinstance(check["ok"], bool)
        assert isinstance(check["detail"], str)


def test_summarize_all_pass():
    checks = [{"name": "a", "ok": True, "detail": ""}]
    assert "全部 1 项通过" in summarize(checks)


def test_summarize_with_failure():
    checks = [
        {"name": "a", "ok": True, "detail": ""},
        {"name": "b", "ok": False, "detail": ""},
    ]
    assert "2 项中 1 项异常" in summarize(checks)
