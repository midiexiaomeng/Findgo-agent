"""ChatClient 测试：验证 DeepSeek V4 Flash 等 OpenAI 兼容接口的解析与降级。"""

from __future__ import annotations

import pytest

from imgsearch_agent.llm.client import ChatClient


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def make_client(**overrides):
    cfg = {
        "backend": "openai",
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-test",
    }
    cfg.update(overrides)
    return ChatClient(cfg)


def test_parses_content(monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: FakeResp(200, {"choices": [{"message": {"content": "hello"}}]}),
    )
    assert make_client().chat([{"role": "user", "content": "hi"}]) == "hello"


def test_falls_back_to_reasoning_content(monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: FakeResp(
            200, {"choices": [{"message": {"content": "", "reasoning_content": "思考结果"}}]}
        ),
    )
    assert make_client().chat([{"role": "user", "content": "hi"}]) == "思考结果"


def test_json_mode_falls_back_when_rejected(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(dict(json or {}))
        if "response_format" in (json or {}):
            return FakeResp(400, {}, "response_format is not supported")
        return FakeResp(200, {"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr("requests.post", fake_post)
    result = make_client().chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert result == "{}"
    assert len(calls) == 2
    assert "response_format" not in calls[1]


def test_reasoning_effort_and_extra_body_passthrough(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json or {})
        return FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("requests.post", fake_post)
    client = make_client(reasoning_effort="low", extra_body={"foo": "bar"})
    client.chat([{"role": "user", "content": "hi"}])
    assert captured["reasoning_effort"] == "low"
    assert captured["foo"] == "bar"
    assert captured["model"] == "deepseek-flash"


def test_empty_content_raises(monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: FakeResp(200, {"choices": [{"message": {"content": ""}}]}),
    )
    with pytest.raises(RuntimeError):
        make_client().chat([{"role": "user", "content": "hi"}])


def test_base_url_without_v1(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        return FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("requests.post", fake_post)
    client = make_client(base_url="https://api.deepseek.com")
    client.chat([{"role": "user", "content": "hi"}])
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
