"""风险标签、失败策略、路径缓存测试。"""

from __future__ import annotations

import inspect

from imgsearch_agent.agent.path_extractor import PathExtractor
from imgsearch_agent.agent.planner import assess_risk
from imgsearch_agent.agent.orchestrator import ImageSearchAgent
from imgsearch_agent.utils.schemas import TaskSpec


def test_assess_risk_high():
    for text in ("删除重复图片", "清理这个目录", "移除索引", "purge trash"):
        assert assess_risk(text) == "high"


def test_assess_risk_normal():
    for text in ("找两张猫", "帮我看看海边的日落"):
        assert assess_risk(text) == "normal"


def test_task_spec_risk_default():
    assert TaskSpec().risk == "normal"


def test_run_tasks_has_failure_strategy():
    params = inspect.signature(ImageSearchAgent.run_tasks).parameters
    assert "continue_on_failure" in params
    assert params["continue_on_failure"].default is True


def test_path_cache_hits_within_ttl(tmp_path):
    extractor = PathExtractor()
    PathExtractor._exists_cache.clear()

    folder = tmp_path / "photos"
    folder.mkdir()
    assert extractor._exists_dir(folder) is True

    folder.rmdir()
    # 缓存未过期，仍认为存在
    assert extractor._exists_dir(folder) is True


def test_path_cache_expires(tmp_path):
    extractor = PathExtractor()
    PathExtractor._exists_cache.clear()

    folder = tmp_path / "photos"
    folder.mkdir()
    extractor._exists_dir(folder)

    # 把时间拨到 TTL 之外
    key = str(folder).lower()
    ts, value = PathExtractor._exists_cache[key]
    PathExtractor._exists_cache[key] = (ts - 60, value)

    folder.rmdir()
    assert extractor._exists_dir(folder) is False
