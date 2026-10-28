"""动态工具留存推荐：规则驱动，只给建议，绝不自动保存。

打分维度：
- 跨会话调用次数
- 历史执行成功率
- 是否硬编码具体路径
- 风险标签

首次生成的工具没有 trace，默认不推荐。
"""

from __future__ import annotations

import re
from typing import Any

_HARDCODED = re.compile(r"[A-Za-z]:[\\/]|/home/|/Users/|/mnt/")


def has_hardcoded_path(code: str) -> bool:
    return bool(_HARDCODED.search(code or ""))


def score_dynamic_tool(spec: dict, traces: list[dict]) -> dict[str, Any]:
    """返回 {score, recommend, reasons}。"""
    name = spec.get("name", "")
    calls = len(traces)
    sessions = len({t.get("session_id") for t in traces if t.get("session_id")})
    failures = sum(1 for t in traces if t.get("error"))
    hardcoded = has_hardcoded_path(spec.get("run_func_code", ""))
    risky = "high-risk" in (spec.get("tags") or [])

    reasons: list[str] = []

    if calls == 0:
        return {
            "score": 0,
            "recommend": False,
            "reasons": ["首次生成，没有历史调用记录"],
        }

    score = 0
    if calls >= 5:
        score += 2
        reasons.append(f"已调用 {calls} 次")
    elif calls >= 2:
        score += 1
        reasons.append(f"已调用 {calls} 次")

    if sessions >= 2:
        score += 2
        reasons.append(f"跨 {sessions} 个会话使用")
    elif sessions == 1:
        score += 1

    if hardcoded:
        score -= 3
        reasons.append("源码里出现硬编码路径，不推荐保存")
    else:
        score += 2
        reasons.append("路径全部从参数传入")

    if risky:
        score -= 2
        reasons.append("带 high-risk 标签")
    else:
        score += 1

    if failures == 0:
        score += 1
        reasons.append("历史执行全部成功")
    else:
        score -= 2
        reasons.append(f"{failures} 次执行失败")

    # 调用次数太少（<3）说明还没被验证过，不推荐
    recommend = calls >= 3 and score >= 4 and not hardcoded and not risky

    return {
        "name": name,
        "score": score,
        "recommend": recommend,
        "reasons": reasons,
    }
