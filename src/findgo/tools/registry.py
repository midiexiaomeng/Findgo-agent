"""工具构造注册表。

`configs/tools.yaml` 决定「有哪些工具、是否启用」，这里决定「怎么把它们建出来」。
构造需要 pipeline 与 sandbox 这类运行时依赖，yaml 表达不了，所以拆成两层：

- yaml：名字、启用开关、给 Planner 看的描述
- 本文件：名字 -> 构造函数

两层必须一一对应。新增工具要同时改两处。漏改时启动直接报错，
而不是静默少一个工具；一致性由 ``tests/test_tool_registry.py`` 守住。
"""

from __future__ import annotations

from typing import Callable

from .base_tool import BaseTool
from .count_indexed_images import CountIndexedImagesTool
from .describe_image import DescribeImageTool
from .export_results import ExportResultsTool
from .filter_results import FilterResultsTool
from .find_duplicates import FindDuplicatesTool
from .folder_scan import FolderScanTool
from .get_index_stats import GetIndexStatsTool
from .image_retrieval import ImageRetrievalTool
from .image_search_by_image import ImageSearchByImageTool
from .index_folder import IndexFolderTool
from .list_indexed_folders import ListIndexedFoldersTool
from .preview_staged import PreviewStagedTool
from .remove_index import RemoveIndexTool
from .switch_vector_backend import SwitchVectorBackendTool
from .trash_manage import TrashManageTool
from .web_image_search import WebImageSearchTool

#: 构造签名统一为 (pipeline, guard)，不用的参数直接忽略
ToolBuilder = Callable[[object, object], BaseTool]

TOOL_BUILDERS: dict[str, ToolBuilder] = {
    "image_retrieval": lambda pipeline, guard: ImageRetrievalTool(pipeline, guard=guard),
    "image_search_by_image": lambda pipeline, guard: ImageSearchByImageTool(pipeline, guard=guard),
    "index_folder": lambda pipeline, guard: IndexFolderTool(pipeline, guard=guard),
    "get_index_stats": lambda pipeline, guard: GetIndexStatsTool(pipeline),
    "switch_vector_backend": lambda pipeline, guard: SwitchVectorBackendTool(pipeline),
    "list_indexed_folders": lambda pipeline, guard: ListIndexedFoldersTool(pipeline),
    "count_indexed_images": lambda pipeline, guard: CountIndexedImagesTool(pipeline, guard=guard),
    "filter_results": lambda pipeline, guard: FilterResultsTool(pipeline),
    "preview_staged": lambda pipeline, guard: PreviewStagedTool(),
    "describe_image": lambda pipeline, guard: DescribeImageTool(pipeline, guard=guard),
    "export_results": lambda pipeline, guard: ExportResultsTool(pipeline, guard=guard),
    "find_duplicates": lambda pipeline, guard: FindDuplicatesTool(pipeline, guard=guard),
    "remove_index": lambda pipeline, guard: RemoveIndexTool(pipeline, guard=guard),
    "trash_manage": lambda pipeline, guard: TrashManageTool(pipeline, guard=guard),
    "web_image_search": lambda pipeline, guard: WebImageSearchTool(pipeline),
    "folder_scan": lambda pipeline, guard: FolderScanTool(guard=guard),
}


def build_tools(
    pipeline,
    guard,
    names: list[str] | None = None,
) -> dict[str, BaseTool]:
    """按名字建工具。``names`` 为空表示全部。

    声明了但没实现构造，或实现了但没声明，都是配置错误，直接抛出来，
    避免出现「yaml 里有、实际调不到」的静默缺功能。
    """
    selected = list(TOOL_BUILDERS) if names is None else list(names)
    missing = [n for n in selected if n not in TOOL_BUILDERS]
    if missing:
        raise ValueError("configs/tools.yaml 里的工具没有对应构造器：" + ", ".join(missing))
    return {name: TOOL_BUILDERS[name](pipeline, guard) for name in selected}
