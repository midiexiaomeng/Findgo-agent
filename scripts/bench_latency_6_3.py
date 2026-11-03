"""6.3 冷启动与热启动耗时基准（论文 6.3 的取数脚本）。

论文 6.3 现在写的是「一次完整检索（含模型首次加载与语义重排）约 29 秒」，
把权重加载和真实检索混在同一个数里。答辩时容易被追问「测了几次、
是单次还是平均、什么硬件」。这个脚本把三段拆开：

    1. 构建 Pipeline（读配置、开向量库）
    2. CLIP 权重加载（warmup）
    3. 首次查询（含查询编码，冷）
    4. 热查询 N 次：重排开 / 重排关

每一步单独计时，热查询取中位数，结果同时打印到屏幕并写到
outputs/latency_6_3.md，可以直接复制成论文里的表。

用法：
    python scripts/bench_latency_6_3.py
    python scripts/bench_latency_6_3.py --query "雪山下的湖泊" --runs 5
    python scripts/bench_latency_6_3.py --config config.yaml --out outputs/latency_6_3.md

注意：测之前先把索引建好（界面里「建立索引」或 cli.py index），
否则搜索会返回 0 条，耗时没有意义。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_T_SCRIPT_START = time.perf_counter()


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _fmt(seconds: float) -> str:
    return f"{seconds:.3f} s"


def main() -> int:
    ap = argparse.ArgumentParser(description="6.3 冷/热启动耗时基准")
    ap.add_argument("--query", default="雪山下的湖泊", help="用于计时的查询词")
    ap.add_argument("--runs", type=int, default=3, help="热查询重复次数，中位数报告")
    ap.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    ap.add_argument("--out", default="outputs/latency_6_3.md", help="结果输出路径")
    args = ap.parse_args()

    from findgo.retriever.pipeline import build_pipeline

    # 1. 构建 Pipeline
    t0 = time.perf_counter()
    pipe = build_pipeline(args.config)
    t_build = time.perf_counter() - t0

    # 2. CLIP 权重加载
    t0 = time.perf_counter()
    try:
        pipe.warmup()
        t_warmup = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 - 加载失败也要把已知耗时打出来
        print(f"[warn] CLIP 预热失败：{exc}", file=sys.stderr)
        t_warmup = float("nan")

    indexed = 0
    try:
        indexed = int(pipe.store.count())
    except Exception:  # noqa: BLE001
        pass

    if indexed == 0:
        print("[warn] 向量库为空，耗时数据不可用。先建索引再跑。", file=sys.stderr)

    device = str(getattr(pipe.encoder, "device", "unknown"))
    rerank_model = getattr(pipe.cfg.rerank, "model", "unknown")

    # 3. 首次查询（冷）
    t0 = time.perf_counter()
    try:
        first = pipe.search_text(args.query, use_rerank=True)
        t_first = time.perf_counter() - t0
        cold_hits = len(first.results)
        cold_recall = first.recall_count
        rerank_used = bool(getattr(first.report, "used", False))
    except Exception as exc:  # noqa: BLE001
        print(f"[error] 首次查询失败：{exc}", file=sys.stderr)
        pipe.close()
        return 1

    # 4. 热查询
    warm_on: list[float] = []
    for _ in range(max(1, args.runs)):
        warm_on.append(pipe.search_text(args.query, use_rerank=True).elapsed)

    warm_off: list[float] = []
    for _ in range(max(1, args.runs)):
        warm_off.append(pipe.search_text(args.query, use_rerank=False).elapsed)

    t_script = time.perf_counter() - _T_SCRIPT_START

    pipe.close()

    rows = [
        ("构建 Pipeline（配置 + 向量库）", _fmt(t_build)),
        ("CLIP 权重加载（warmup）", _fmt(t_warmup)),
        ("首次查询（冷，含查询编码）", _fmt(t_first)),
        (f"热查询 · 重排开（{args.runs} 次中位数）", _fmt(_median(warm_on))),
        (f"热查询 · 重排关（{args.runs} 次中位数）", _fmt(_median(warm_off))),
        ("脚本启动 → 首个结果（端到端冷启动）", _fmt(t_script)),
    ]

    lines = [
        "# 6.3 冷启动与热启动耗时实测",
        "",
        f"- 查询词：{args.query}",
        f"- 索引条数：{indexed}",
        f"- 计算设备：{device}",
        f"- 重排模型：{rerank_model}（本次冷查询重排{'已生效' if rerank_used else '未生效，读数是纯向量口径'}）",
        f"- 冷查询返回：召回 {cold_recall} 条，返回 {cold_hits} 张",
        "",
        "| 阶段 | 耗时 |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines += [
        "",
        "热查询逐次读数（秒）：重排开 "
        + "、".join(f"{v:.3f}" for v in warm_on)
        + "；重排关 "
        + "、".join(f"{v:.3f}" for v in warm_off),
        "",
        "论文 6.3 的写法：把「首次查询」与「热查询」分开报，注明重复次数与取中位数，"
        "再标设备。原来那句「含模型首次加载约 29 秒」用端到端冷启动那一行替换，"
        "不要再用一个笼统的 29 秒。",
        "",
    ]
    table = "\n".join(lines)

    print(table)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")
    print(f"[ok] 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
