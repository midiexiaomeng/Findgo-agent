"""检索质量评估与基准测试。

把"能搜到图"变成可量化的指标：Recall@K、mRR、mAP、NDCG@K，
并对比"纯向量检索"与"向量+大模型重排"两种策略。这是项目区别于
一般 demo 的核心，也是论文实验章节的数据来源。

用法：
    # 1. 准备 COCO 标注（captions_val2017.json，约 241MB）
    #    下载 annotations_trainval2017.zip 并解压，或在有网时运行：
    #    python scripts/download_models.py   （此脚本只下模型，需另下标注）
    #    简便方式：从 https://images.cocodataset.org/annotations/annotations_trainval2017.zip 下载

    # 2. 运行评估
    python scripts/evaluate.py --captions data/captions_val2017.json --n 100

说明：
    - 查询取自 COCO 的英文 caption，目标为对应的图片。Chinese-CLIP 对
      英文查询支持有限，因此 Recall 会偏低，这本身是一个可写入论文的
      实验发现，也是引入中文描述的动机。
    - 重排较慢，默认只评估前 --rerank-n 条查询做重排，其余只做纯向量。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_indexed_ids(manifest_path: Path) -> set[int]:
    """从 manifest.json 读取已索引图片的 COCO image_id。"""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids: set[int] = set()
    for img in data.get("images", []):
        stem = Path(img["file"]).stem
        if stem.isdigit():
            ids.add(int(stem))
    return ids


def load_captions(path: Path) -> dict[int, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    caps: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        caps[int(ann["image_id"])].append(str(ann.get("caption", "")).strip())
    return caps


def build_queries(indexed_ids, captions, n, seed):
    rng = random.Random(seed)
    candidates = [i for i in sorted(indexed_ids) if i in captions and captions[i]]
    if n > len(candidates):
        n = len(candidates)
    picked = rng.sample(candidates, n)
    return [(captions[i][0], i) for i in picked]  # 用第一条 caption 作查询


def evaluate(searcher, queries, top_k, recall_k, use_rerank):
    ks = sorted({1, 5, top_k})
    found = {k: 0 for k in ks}
    rr_sum = 0.0
    ap_sum = 0.0
    ndcg_sum = 0.0
    elapsed = 0.0
    ranks: list[int | None] = []

    for idx, (query, target_id) in enumerate(queries, 1):
        target_file = f"{target_id:012d}.jpg"
        t0 = time.time()
        resp = searcher.search_text(
            query, top_k=top_k, recall_k=recall_k, use_rerank=use_rerank
        )
        elapsed += time.time() - t0

        rank = None
        for r in resp.results:
            if r.filename == target_file:
                rank = r.rank
                break

        if rank is not None:
            ranks.append(rank)
            for k in ks:
                if rank <= k:
                    found[k] += 1
            rr_sum += 1.0 / rank
            ap_sum += 1.0 / rank  # 单 ground truth 时 AP = 1/rank
            if rank <= top_k:
                ndcg_sum += 1.0 / math.log2(rank + 1)
        else:
            ranks.append(None)

        if idx % 20 == 0:
            print(f"  {idx}/{len(queries)}")

    n = len(queries)
    return {
        "n": n,
        **{f"recall@{k}": found[k] / n for k in ks},
        "mrr": rr_sum / n,
        "map": ap_sum / n,
        f"ndcg@{top_k}": ndcg_sum / n,
        "avg_ms": (elapsed / n) * 1000,
    }


def fmt_metrics(m: dict) -> str:
    lines = [f"查询数   : {m['n']}"]
    for k in sorted(m):
        if k in ("n", "avg_ms"):
            continue
        lines.append(f"{k:<8}: {m[k]:.4f}")
    lines.append(f"平均耗时 : {m['avg_ms']:.0f} ms/条")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评估")
    parser.add_argument("--captions", required=True, help="COCO captions_val2017.json 路径")
    parser.add_argument("--manifest", default=str(ROOT / "datasets/coco_2000/manifest.json"))
    parser.add_argument("--n", type=int, default=100, help="评估查询总数")
    parser.add_argument("--rerank-n", type=int, default=20, help="做重排评估的查询数（较慢）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--recall-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from imgsearch import Config
    from imgsearch.pipeline import Pipeline

    caps_path = Path(args.captions)
    if not caps_path.exists():
        print(f"标注文件不存在: {caps_path}")
        print("请先下载 COCO annotations_trainval2017.zip 并解压出 captions_val2017.json")
        return 1

    cfg = Config.from_yaml()
    pipe = Pipeline(cfg)

    captions = load_captions(caps_path)
    indexed = load_indexed_ids(Path(args.manifest))
    queries = build_queries(indexed, captions, args.n, args.seed)
    if not queries:
        print("没有可评估的查询：请确认 manifest.json 与 captions 的 image_id 能对上")
        pipe.close()
        return 1
    print(f"共 {len(queries)} 条评估查询")

    print("\n== 纯向量检索 ==")
    vec = evaluate(pipe.searcher, queries, args.top_k, args.recall_k, use_rerank=False)

    rer = None
    rerank_queries = queries[: max(1, args.rerank_n)]
    print(f"\n== 向量 + 大模型重排（{len(rerank_queries)} 条）==")
    rer = evaluate(pipe.searcher, rerank_queries, args.top_k, args.recall_k, use_rerank=True)

    print("\n" + "=" * 50)
    print("纯向量检索")
    print(fmt_metrics(vec))
    if rer:
        print("\n向量 + 大模型重排")
        print(fmt_metrics(rer))
        print("\n差异（重排 - 纯向量，在重排子集上对比）")
        for k in sorted(vec):
            if k not in ("n", "avg_ms") and k in rer:
                print(f"{k:<8}: {rer[k] - vec[k]:+.4f}")

    out = {"vec": vec, "rerank": rer, "rerank_n": len(rerank_queries) if rer else 0}
    report_path = ROOT / "data" / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
