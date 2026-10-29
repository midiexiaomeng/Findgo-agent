"""Critic 多维度校验与错误分类测试。"""

from __future__ import annotations

import pytest
from findgo.agent.critic import Critic
from findgo.agent.executor import ToolExecutor
from findgo.utils.schemas import TaskSpec


def _result(images=None, confidence=0.5, **extra):
    return {
        "images": images if images is not None else [{"path": "a.jpg", "caption": "white cat"}],
        "confidence": confidence,
        **extra,
    }


def _task(top_k=0, must=None):
    return TaskSpec(
        query="猫",
        top_k=top_k,
        constraints={"must": must or []},
    )


# ── 基础 ──────────────────────────────────────────────────────────────────


def test_empty_result_fails():
    c = Critic(threshold=0.25)
    critique = c.check(_result(images=[]))
    assert critique.passed is False
    assert "零结果" in critique.issues
    assert critique.retryable is True


def test_good_result_passes():
    c = Critic(threshold=0.25)
    critique = c.check(_result(confidence=0.6), _task(top_k=1))
    assert critique.passed is True
    assert critique.severity == "ok"


# ── 数量维度 ──────────────────────────────────────────────────────────────


def test_count_gap_detected():
    c = Critic(threshold=0.25)
    images = [{"path": f"{i}.jpg", "caption": "white cat"} for i in range(2)]
    critique = c.check(_result(images=images, confidence=0.6), _task(top_k=5))

    assert critique.passed is False
    assert critique.missing_count == 3
    assert any("差 3 张" in i for i in critique.issues)
    assert critique.retryable is True


def test_count_within_tolerance():
    """差 1 张不算问题。"""
    c = Critic(threshold=0.25)
    images = [{"path": f"{i}.jpg", "caption": "white cat"} for i in range(4)]
    critique = c.check(_result(images=images, confidence=0.6), _task(top_k=5))
    assert critique.passed is True


def test_no_count_requirement():
    c = Critic(threshold=0.25)
    critique = c.check(_result(confidence=0.6), _task(top_k=0))
    assert critique.passed is True


# ── 约束维度 ──────────────────────────────────────────────────────────────


def test_must_violation_detected():
    c = Critic(threshold=0.25)
    images = [{"path": "a.jpg", "caption": "a black cat sitting"}]
    critique = c.check(_result(images=images, confidence=0.7), _task(must=["白色"]))

    assert critique.passed is False
    assert "白色" in critique.violated
    assert any("必需条件" in i for i in critique.issues)


def test_must_satisfied():
    c = Critic(threshold=0.25)
    images = [{"path": "a.jpg", "caption": "a white cat lying"}]
    critique = c.check(_result(images=images, confidence=0.7), _task(must=["白色"]))
    assert critique.passed is True


def test_must_ignored_without_caption():
    """没有描述时无法判断约束，不应该判失败。"""
    c = Critic(threshold=0.25)
    images = [{"path": "a.jpg", "caption": ""}]
    critique = c.check(_result(images=images, confidence=0.7), _task(must=["白色"]))
    assert critique.passed is True


def test_constraint_violation_not_retryable():
    """约束不命中，改写查询救不回来。"""
    c = Critic(threshold=0.25)
    images = [{"path": "a.jpg", "caption": "a black cat sitting"}]
    critique = c.check(_result(images=images, confidence=0.7), _task(must=["白色"]))
    assert critique.retryable is False
    assert critique.severity == "fatal"


# ── 置信维度 ──────────────────────────────────────────────────────────────


def test_low_confidence_detected():
    c = Critic(threshold=0.3)
    critique = c.check(_result(confidence=0.1))
    assert critique.passed is False
    assert any("低于阈值" in i for i in critique.issues)


def test_multiple_issues_collected():
    c = Critic(threshold=0.5)
    images = [{"path": "a.jpg", "caption": "black cat"}]
    critique = c.check(_result(images=images, confidence=0.1), _task(top_k=5, must=["白色"]))

    assert len(critique.issues) >= 3
    assert critique.missing_count == 4
    assert "白色" in critique.violated


# ── 错误分类 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "error,expect",
    [
        (FileNotFoundError("x"), "io"),
        (PermissionError("x"), "io"),
        (IsADirectoryError("x"), "io"),
        (MemoryError(), "memory"),
        (RuntimeError("CUDA out of memory"), "gpu_oom"),
        (RuntimeError("VectorStoreError: locked"), "vectorstore"),
        (ConnectionError("connection refused"), "network"),
        (TimeoutError("timed out"), "network"),
        (ValueError("misc"), "unknown"),
    ],
)
def test_error_classification(error, expect):
    category, hint = ToolExecutor._classify_error(error)
    assert category == expect
    assert hint


def test_error_hint_is_actionable():
    _, hint = ToolExecutor._classify_error(RuntimeError("CUDA out of memory"))
    assert "CPU" in hint or "显存" in hint


# ── 修复提示传递 ──────────────────────────────────────────────────────────


def test_hint_surfaces_in_reply():
    import inspect

    from findgo.agent.orchestrator import FindgoAgent

    src = inspect.getsource(FindgoAgent.run_tasks)
    assert 'result.get("hint"' in src
    assert "建议：" in src


def test_oom_not_retried():
    import inspect

    from findgo.agent.orchestrator import FindgoAgent

    src = inspect.getsource(FindgoAgent._execute_with_retry)
    assert "gpu_oom" in src
    assert "不重试" in src
