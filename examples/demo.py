"""最小可运行示例。

    python examples/demo.py                    # 用默认查询
    python examples/demo.py "雪山和湖"          # 指定查询
    python examples/demo.py --index examples/test_images

先建索引再搜索，完整走一遍三级链路。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from imgsearch_agent.retriever import build_pipeline  # noqa: E402


def main() -> int:
    args = sys.argv[1:]

    pipe = build_pipeline()
    print(f"向量库: {pipe.store.backend_name} | 现有 {pipe.store.count()} 张")

    if args and args[0] == "--index":
        folder = args[1] if len(args) > 1 else "examples/test_images"
        stats = pipe.index_folder(folder, recursive=True)
        print(stats.summary())
        args = args[2:]

    if pipe.store.count() == 0:
        print("库是空的，先建索引：python examples/demo.py --index examples/test_images")
        pipe.close()
        return 1

    query = args[0] if args else "夕阳下的海边"
    print(f"\n查询: {query}\n")

    response = pipe.search_text(query, top_k=5)
    print(response.status_line())
    print("-" * 60)
    for r in response.results:
        llm = "-" if r.llm_score is None else f"{r.llm_score:.2f}"
        print(f"[{r.rank}] 融合 {r.score:.3f} | 向量 {r.vec_score:.3f} | LLM {llm}")
        print(f"    {r.path}")
        if r.caption:
            print(f"    {r.caption}")
    print("-" * 60)

    pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
