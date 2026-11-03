"""以图搜图工具的封装测试：路径守卫、结果结构与参数校验。

检索质量由 tests/test_searcher.py 和 tests/test_index_search.py 负责，
这里只验证工具层把路径、scope、分数转成 Agent 能吃的结构。
"""

from __future__ import annotations

from findgo.tools.image_search_by_image import ImageSearchByImageTool
from findgo.utils.exceptions import ValidationError
from findgo.utils.path_guard import PathGuard


class FakeItem:
    def __init__(self, rank: int, path: str, caption: str, score: float) -> None:
        self.rank = rank
        self.path = path
        self.caption = caption
        self.score = score


class FakeResp:
    def __init__(self, items) -> None:
        self.results = list(items)

    @staticmethod
    def status_line() -> str:
        return "召回 50 条 · 返回 2 张"


class FakeSearcher:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_image(self, image_path, top_k=8, scope=None):
        self.calls.append({"image_path": image_path, "top_k": top_k, "scope": scope})
        return FakeResp(
            [
                FakeItem(1, "D:/photos/cat.jpg", "a cat", 0.9),
                FakeItem(2, "D:/photos/kitten.jpg", "a kitten", 0.6),
            ]
        )


class FakePipeline:
    def __init__(self) -> None:
        self.searcher = FakeSearcher()


def make_tool(tmp_path, pipeline=None):
    pipeline = pipeline or FakePipeline()
    return (
        ImageSearchByImageTool(pipeline, guard=PathGuard(allowed_roots=[str(tmp_path)])),
        pipeline,
    )


def make_image(tmp_path, name: str = "query.jpg"):
    path = tmp_path / name
    path.write_bytes(b"not-a-real-jpeg")
    return path


def test_run_returns_structured_result(tmp_path):
    tool, pipeline = make_tool(tmp_path)
    query = make_image(tmp_path)

    out = tool.run(str(query), top_k=5)

    assert out["query"] == str(query)
    assert [i["rank"] for i in out["images"]] == [1, 2]
    assert out["images"][0]["path"] == "D:/photos/cat.jpg"
    assert out["images"][0]["score"] == 0.9
    assert out["confidence"] == 0.75
    assert out["summary"] == "召回 50 条 · 返回 2 张"

    call = pipeline.searcher.calls[0]
    assert call["image_path"] == str(query.resolve())
    assert call["top_k"] == 5
    assert call["scope"] is None


def test_folder_path_becomes_scope(tmp_path):
    tool, pipeline = make_tool(tmp_path)
    query = make_image(tmp_path)
    folder = tmp_path / "photos"
    folder.mkdir()

    tool.run(str(query), folder_path=str(folder))

    assert pipeline.searcher.calls[0]["scope"] == str(folder.resolve())


def test_non_file_is_rejected_before_search(tmp_path):
    tool, pipeline = make_tool(tmp_path)
    missing = tmp_path / "missing.jpg"

    out = tool.run(str(missing))

    assert out["images"] == []
    assert out["confidence"] == 0.0
    assert "不是文件" in out["summary"]
    assert pipeline.searcher.calls == []


def test_unsafe_image_path_is_rejected(tmp_path):
    tool, pipeline = make_tool(tmp_path)

    out = tool.run("C:/Windows/System32/drivers/etc/hosts")

    assert out["images"] == []
    assert "不安全" in out["summary"]
    assert pipeline.searcher.calls == []


def test_unsafe_folder_path_is_rejected(tmp_path):
    tool, pipeline = make_tool(tmp_path)
    query = make_image(tmp_path)

    out = tool.run(str(query), folder_path="C:/Windows")

    assert out["images"] == []
    assert "不安全" in out["summary"]
    assert pipeline.searcher.calls == []


def test_args_schema_defaults_and_bounds(tmp_path):
    args = ImageSearchByImageTool.args_schema(image_path="a.jpg")
    assert args.top_k == 8
    assert args.folder_path == ""

    tool, _ = make_tool(tmp_path)
    try:
        tool.validate({"image_path": "a.jpg", "top_k": 0})
    except ValidationError:
        pass
    else:
        raise AssertionError("top_k 越界时应抛出 ValidationError")
