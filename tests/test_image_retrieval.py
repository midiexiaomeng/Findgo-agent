"""图像检索工具测试：验证封装后的返回结构。"""

from __future__ import annotations

from imgsearch_agent.tools.image_retrieval import ImageRetrievalTool


class FakeResult:
    def __init__(self):
        self.rank = 1
        self.path = "D:/photos/cat.jpg"
        self.caption = "a cat sitting on a windowsill"
        self.score = 0.9


class FakeResp:
    def __init__(self):
        self.results = [FakeResult()]

    def status_line(self):
        return "召回 50 条 · 返回 1 张"


class FakeSearcher:
    def search_text(self, query, top_k=8, scope=None, all_results=False, should_stop=None, exclude_paths=None):
        return FakeResp()


class FakePipeline:
    def __init__(self):
        self.searcher = FakeSearcher()


def test_run_returns_structured_result():
    tool = ImageRetrievalTool(FakePipeline())
    out = tool.run("一只猫", top_k=5)

    assert out["query"] == "一只猫"
    assert len(out["images"]) == 1
    assert out["images"][0]["path"] == "D:/photos/cat.jpg"
    assert out["images"][0]["caption"].startswith("a cat")
    assert out["confidence"] == 0.9
    assert "召回" in out["summary"]


def test_args_schema_defaults():
    args = ImageRetrievalTool.args_schema(query="猫")
    assert args.top_k == 8


def test_validate_rejects_missing_query():
    from imgsearch_agent.utils.exceptions import ValidationError

    tool = ImageRetrievalTool(FakePipeline())
    try:
        tool.validate({})
    except ValidationError:
        return
    raise AssertionError("缺少 query 时应抛出 ValidationError")
