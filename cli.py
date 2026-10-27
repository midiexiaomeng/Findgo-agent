"""命令行工具。

    python cli.py index  D:\\Photos --reset
    python cli.py search "夕阳下的海边" --top-k 10
    python cli.py image  D:\\test.jpg
    python cli.py stats
    python cli.py doctor
    python cli.py serve
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from imgsearch_agent.retriever import Config, __version__  # noqa: E402
from imgsearch_agent.retriever.runtime import human_time, setup_logging  # noqa: E402


def _pipeline(args):
    from imgsearch_agent.retriever.pipeline import Pipeline

    cfg = Config.from_yaml(args.config)
    return Pipeline(cfg, backend=getattr(args, "backend", None))


def _print_results(response) -> None:
    print()
    print(f"查询: {response.query}")
    if response.query_caption:
        print(f"查询图描述: {response.query_caption}")
    print(response.status_line())
    if response.note:
        print(f"说明: {response.note}")
    print("-" * 72)

    if not response.results:
        print("没有找到结果")
        return

    for r in response.results:
        llm = "  -  " if r.llm_score is None else f"{r.llm_score:.3f}"
        flag = "" if r.exists else "  [文件已丢失]"
        print(f"[{r.rank:>2}] 融合 {r.score:.3f} | 向量 {r.vec_score:.3f} | LLM {llm}{flag}")
        print(f"     {r.path}")
        if r.caption:
            print(f"     {r.caption}")
        print()

    print("-" * 72)
    print(f"共 {len(response.results)} 条，耗时 {human_time(response.elapsed)}")


# ── 子命令 ────────────────────────────────────────────────────────────────────


def cmd_index(args) -> int:
    p = _pipeline(args)
    print(f"向量库: {p.store.backend_name} | collection={p.store.collection} | 现有 {p.store.count()} 条")

    last = {"pct": -1}

    def progress(done: int, total: int, name: str) -> None:
        pct = int(done * 100 / max(total, 1))
        if pct >= last["pct"] + 5 or done == total:
            last["pct"] = pct
            print(f"  {done}/{total}  ({pct}%)  {name}")

    stats = p.index_folder(
        args.folder,
        recursive=not args.no_recursive,
        reset=args.reset,
        skip_existing=not args.force,
        with_caption=not args.no_caption,
        prune_missing=args.prune,
        progress=progress,
    )
    print()
    print(stats.summary())
    print(f"库内总数: {p.store.count()}")
    if stats.errors:
        print("错误样例:")
        for err in stats.errors[:5]:
            print(f"  - {err}")
    p.close()
    return 0 if stats.failed == 0 else 1


def cmd_search(args) -> int:
    p = _pipeline(args)
    response = p.search_text(
        args.query,
        top_k=args.top_k,
        recall_k=args.recall_k,
        use_rerank=not args.no_rerank,
    )
    if args.json:
        print(json.dumps(response.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_results(response)
    p.close()
    return 0 if response.results else 2


def cmd_image(args) -> int:
    path = Path(args.image)
    if not path.exists():
        print(f"图片不存在: {path}", file=sys.stderr)
        return 1
    p = _pipeline(args)
    response = p.search_image(
        path,
        top_k=args.top_k,
        recall_k=args.recall_k,
        use_rerank=not args.no_rerank,
        exclude_self=not args.include_self,
    )
    if args.json:
        print(json.dumps(response.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_results(response)
    p.close()
    return 0 if response.results else 2


def cmd_stats(args) -> int:
    p = _pipeline(args)
    print(json.dumps(p.stats(), ensure_ascii=False, indent=2))
    p.close()
    return 0


def cmd_serve(args) -> int:
    from webui.app import launch

    launch(server_name=args.host, server_port=args.port, inbrowser=not args.no_browser)
    return 0


def cmd_doctor(args) -> int:
    print(f"imgsearch {__version__}")
    print("=" * 60)

    ok = True
    print(f"Python      : {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("  ! 需要 Python 3.9+")
        ok = False

    for mod, label in [
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("chromadb", "chromadb"),
        ("qdrant_client", "qdrant-client"),
        ("gradio", "gradio"),
        ("huggingface_hub", "huggingface_hub"),
    ]:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{label:<12}: {ver}")
        except Exception as e:
            print(f"{label:<12}: 缺失 ({e})")
            ok = False

    try:
        import torch

        print(f"CUDA        : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                print(
                    f"  GPU {i}     : {prop.name} "
                    f"{prop.total_memory / 1024**3:.1f} GB"
                )
    except Exception:
        pass

    cfg = Config.from_yaml(getattr(args, "config", None))
    print("-" * 60)

    from imgsearch_agent.retriever.runtime import resolve_model_source

    clip_src = resolve_model_source(
        cfg.embedding.model, cfg.models_dir, cfg.embedding.fallback
    )
    blip_src = resolve_model_source(
        cfg.caption.model, cfg.models_dir, cfg.caption.fallback
    )
    print(f"CLIP        : {clip_src}")
    print(f"BLIP        : {blip_src}")
    if not Path(clip_src).exists():
        print("  ! CLIP 未在本地，首次运行会联网下载（或先跑 download_models.py）")
        ok = False
    if not Path(blip_src).exists():
        print("  ! BLIP 未在本地，首次运行会联网下载")

    try:
        from imgsearch_agent.retriever.pipeline import Pipeline

        p = Pipeline(cfg)
        print(
            f"向量库      : {p.store.backend_name} / {p.store.collection} / "
            f"{p.store.count()} 条"
        )
        health = p.health(ttl=0)
        print(f"LLM         : {health['llm_message']}")
        if not health["llm_ok"]:
            print("  ! LLM 不可用，检索会自动退化为纯向量排序")
        p.close()
    except Exception as e:
        print(f"向量库      : 初始化失败 - {e}")
        ok = False

    try:
        usage = shutil.disk_usage(str(cfg.data_dir.parent))
        print(f"磁盘剩余    : {usage.free / 1024**3:.1f} GB")
    except Exception:
        pass

    print("=" * 60)
    print("检查通过" if ok else "存在需要处理的问题，见上方 ! 标记")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imgsearch",
        description="本地多模态 RAG 图像检索（以文搜图 / 以图搜图）",
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--version", action="version", version=f"imgsearch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="为图片文件夹建立索引")
    p_index.add_argument("folder", help="图片文件夹")
    p_index.add_argument("--backend", choices=["chromadb", "qdrant"], default=None)
    p_index.add_argument("--reset", action="store_true", help="先清空当前库")
    p_index.add_argument("--force", action="store_true", help="覆盖已索引的图片")
    p_index.add_argument("--no-recursive", action="store_true", help="不扫描子文件夹")
    p_index.add_argument("--no-caption", action="store_true", help="不生成图像描述")
    p_index.add_argument("--prune", action="store_true", help="清理磁盘上已不存在的记录")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="以文搜图")
    p_search.add_argument("query")
    p_search.add_argument("--backend", choices=["chromadb", "qdrant"], default=None)
    p_search.add_argument("--top-k", type=int, default=None)
    p_search.add_argument("--recall-k", type=int, default=None)
    p_search.add_argument("--no-rerank", action="store_true", help="只做向量检索")
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_image = sub.add_parser("image", help="以图搜图")
    p_image.add_argument("image")
    p_image.add_argument("--backend", choices=["chromadb", "qdrant"], default=None)
    p_image.add_argument("--top-k", type=int, default=None)
    p_image.add_argument("--recall-k", type=int, default=None)
    p_image.add_argument("--no-rerank", action="store_true")
    p_image.add_argument("--include-self", action="store_true", help="结果中包含查询图本身")
    p_image.add_argument("--json", action="store_true")
    p_image.set_defaults(func=cmd_image)

    p_stats = sub.add_parser("stats", help="查看库统计")
    p_stats.add_argument("--backend", choices=["chromadb", "qdrant"], default=None)
    p_stats.set_defaults(func=cmd_stats)

    p_serve = sub.add_parser("serve", help="启动 Web 界面")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--no-browser", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_doc = sub.add_parser("doctor", help="环境自检")
    p_doc.set_defaults(func=cmd_doctor)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging("INFO")
    if args.command == "serve" and args.host is None:
        cfg = Config.from_yaml(args.config)
        args.host = cfg.app.host
        args.port = args.port or cfg.app.port
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
