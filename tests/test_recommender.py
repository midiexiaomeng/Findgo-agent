"""动态工具留存推荐测试。"""

from __future__ import annotations

from findgo.agent.recommender import has_hardcoded_path, score_dynamic_tool

SPEC = {
    "name": "count_jpg",
    "tags": ["read-only"],
    "run_func_code": (
        "def run(self, args):\n" '    path = sandbox_check_path(args["folder"], "read")\n'
    ),
}

HARDCODED_CODE = "def run(self, args):\n" '    path = r"D:\\photos\\fixed"\n'


def traces(n, sessions=1, errors=0):
    return [
        {"session_id": f"s{i % sessions}", "error": "x" if i < errors else None} for i in range(n)
    ]


def test_first_time_not_recommended():
    result = score_dynamic_tool(SPEC, [])
    assert result["recommend"] is False
    assert result["score"] == 0
    assert "首次" in result["reasons"][0]


def test_frequent_cross_session_recommended():
    result = score_dynamic_tool(SPEC, traces(6, sessions=3))
    assert result["recommend"] is True


def test_single_use_not_recommended():
    result = score_dynamic_tool(SPEC, traces(1, sessions=1))
    assert result["recommend"] is False


def test_hardcoded_path_rejected():
    assert has_hardcoded_path(HARDCODED_CODE) is True
    bad = dict(SPEC, run_func_code=HARDCODED_CODE)
    result = score_dynamic_tool(bad, traces(10, sessions=5))
    assert result["recommend"] is False


def test_high_risk_not_recommended():
    risky = dict(SPEC, tags=["high-risk"])
    result = score_dynamic_tool(risky, traces(10, sessions=5))
    assert result["recommend"] is False


def test_failures_reduce_score():
    ok = score_dynamic_tool(SPEC, traces(6, sessions=3))
    bad = score_dynamic_tool(SPEC, traces(6, sessions=3, errors=3))
    assert bad["score"] < ok["score"]
