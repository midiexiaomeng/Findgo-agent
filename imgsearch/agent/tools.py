"""Agent 工具层：把系统现有能力封装成 Agent 可调用的工具。

每个工具是一段带描述的命名函数。Agent 通过读取工具描述来决定
调用哪一个，而不是写死 if-else，因此新增能力只需在这里注册。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..searcher import SearchResult


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def signature(self) -> str:
        params = ", ".join(self.parameters) or "无参数"
        return f"{self.name}({params}) — {self.description}"


def _result_payload(resp) -> dict:
    """把一次检索响应的结果转成便于 Agent 消化和界面展示的结构。"""
    items = []
    for r in resp.results[:8]:
        items.append(
            {
                "rank": r.rank,
                "filename": r.filename,
                "path": r.path,
                "score": round(float(r.score), 3),
                "caption": (r.caption or "")[:80],
            }
        )
    return {
        "results": resp.results[:8],  # 保留 SearchResult 对象给界面直接渲染
        "items": items,
        "summary": resp.status_line(),
    }


def build_tools(pipeline) -> list[Tool]:
    """注册系统对外暴露的工具。"""

    def search_images(query: str, top_k: int = 6) -> dict:
        resp = pipeline.search_text(query, top_k=top_k)
        return _result_payload(resp)

    def search_by_image(path: str, top_k: int = 6) -> dict:
        resp = pipeline.search_image(path, top_k=top_k)
        return _result_payload(resp)

    def get_stats() -> dict:
        s = pipeline.stats()
        return {
            "results": [],
            "items": [],
            "summary": (
                f"图片库共有 {s['images']} 张图片，向量库为 {s['backend']}，"
                f"语义重排{'已启用' if s['rerank_enabled'] else '未启用'}。"
            ),
            "stats": s,
        }

    def index_folder(path: str) -> dict:
        stats = pipeline.index_folder(path)
        return {
            "results": [],
            "items": [],
            "summary": stats.summary(),
        }

    return [
        Tool(
            name="search_images",
            description="用自然语言描述搜索图片，例如“海边的日落”",
            parameters={"query": {"type": "string", "description": "图片内容描述"},
                        "top_k": {"type": "integer", "description": "返回数量，默认 6"}},
            func=search_images,
        ),
        Tool(
            name="search_by_image",
            description="用一张本地图片搜索相似图片",
            parameters={"path": {"type": "string", "description": "本地图片的绝对路径"},
                        "top_k": {"type": "integer", "description": "返回数量，默认 6"}},
            func=search_by_image,
        ),
        Tool(
            name="get_stats",
            description="查询图片库的统计信息",
            parameters={},
            func=get_stats,
        ),
        Tool(
            name="index_folder",
            description="扫描并索引一个文件夹里的图片",
            parameters={"path": {"type": "string", "description": "要索引的文件夹路径"}},
            func=index_folder,
        ),
    ]


def tools_to_prompt(tools: list[Tool]) -> str:
    return "\n".join(f"{i + 1}. {t.signature()}" for i, t in enumerate(tools))
