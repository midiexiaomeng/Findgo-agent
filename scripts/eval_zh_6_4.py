"""6.4 检索效果实验：中文查询 + A/B 对比（论文 6.4 的取数脚本）。

为什么另写一个而不是直接用 scripts/evaluate.py：
evaluate.py 直接拿 COCO 的英文 caption 当查询，而本项目用的是
Chinese-CLIP，英文查询本身就会压低召回（evaluate.py 的说明里也承认这点）。
论文 6.4 要的是中文查询，所以这里先用本地/云端 LLM 把 caption 翻成中文，
缓存成 queries_zh.json，再跑「纯向量」与「向量 + 重排」两组，指标按
Recall@1/5/10 与 MRR 输出，并生成可以直接粘进论文的 Markdown 表格。

前提：
    1. 索引已经建好（向量库里得有图），否则召回全空。
    2. 有 COCO 的 captions_val2017.json（annotations_trainval2017.zip 里的
       那一个文件），用于取被索引图片的官方描述。
       不想下那个 241MB 的包也可以：加 --from-store，直接用索引阶段 BLIP
       写进向量库 caption 字段的英文描述来造查询（口径变成「模型描述」，
       不是人工标注，论文里要写明）。
    3. LLM 可用（config.yaml 里 rerank 后端，DeepSeek 需 DEEPSEEK_API_KEY，
       或 ollama 本地服务），只用来做中文改写。翻译结果会缓存，可重复跑。

用法：
    # 第一次：翻译 + 评估，200 条查询
    python scripts/eval_zh_6_4.py --captions D:/path/captions_val2017.json --n 200

    # 已翻译过：跳过翻译直接用缓存
    python scripts/eval_zh_6_4.py --reuse

    # 重排只跑前 40 条（重排慢，40 条足够出趋势）
    python scripts/eval_zh_6_4.py --reuse --rerank-n 40

输出：
    datasets/coco_2000/queries_zh.json   中文查询集（缓存）
    outputs/eval_zh_6_4.md               可直接粘进 6.4 的表
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

QUERY_CACHE = ROOT / "datasets" / "coco_2000" / "queries_zh.json"
DEFAULT_MANIFEST = ROOT / "datasets" / "coco_2000" / "manifest.json"


# ── 数据准备 ────────────────────────────────────────────────────────────────


def load_indexed_ids(manifest_path: Path) -> set[int]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids: set[int] = set()
    for img in data.get("images", []):
        stem = Path(str(img.get("file", ""))).stem
        if stem.isdigit():
            ids.add(int(stem))
    return ids


def load_pairs_from_store(store) -> list[tuple[str, str]]:
    """从向量库取 (英文描述, 文件名)，不需要下载 COCO 标注。

    索引阶段 BLIP 生成的描述已经写在每条记录的 caption 字段里（见
    retriever/indexer.py 的 meta["caption"]），与 data/captions.sqlite
    同源，但多了 filename，正好可以构造「描述 → 原图」的查询—目标对。
    代价是查询源是模型自己写的描述，不是人工标注，口径要在论文里说明。
    """
    ids = list(store.get_all_ids())
    pairs: list[tuple[str, str]] = []
    step = 512
    for start in range(0, len(ids), step):
        for hit in store.get(ids[start : start + step]):
            meta = getattr(hit, "metadata", {}) or {}
            cap = str(meta.get("caption", "") or "").strip()
            name = str(meta.get("filename", "") or "").strip()
            if cap and name:
                pairs.append((cap, name))
    return pairs


def build_queries_from_store(pipe, n: int, seed: int, batch: int) -> list[dict]:
    """用向量库里已有的英文描述构造中文查询集（不依赖 COCO 标注文件）。"""
    from findgo.retriever.config import Config
    from findgo.retriever.llm.client import LLMClient

    pool = load_pairs_from_store(pipe.store)
    if not pool:
        raise RuntimeError(
            "向量库里没有可用的 caption 字段，先建一次带描述的索引（caption.enabled: true）"
        )
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n, len(pool)))

    # 优先复用 Pipeline 里那个客户端：transformers 后端下再新建一个实例
    # 会把 3B 权重载入两次，8GB 显存直接不够用
    client = getattr(getattr(pipe, "reranker", None), "client", None)
    if client is None:
        client = LLMClient(Config.from_yaml())
    ok, why = client.is_available()
    if not ok:
        raise RuntimeError(f"翻译需要可用的 LLM：{why}")

    items: list[dict] = []
    for start in range(0, len(picked), batch):
        chunk = picked[start : start + batch]
        zh = translate_batch(client, chunk)
        for (cap, name), query in zip(chunk, zh):
            items.append({"query": query, "filename": name, "caption": cap})
        print(f"  翻译 {min(start + batch, len(picked))}/{len(picked)}", flush=True)
        QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        QUERY_CACHE.write_text(
            json.dumps(
                {"seed": seed, "n": len(items), "source": "store", "items": items},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return items


def load_captions(path: Path) -> dict[int, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    caps: dict[int, list[str]] = defaultdict(list)
    for ann in data.get("annotations", []):
        text = str(ann.get("caption", "")).strip()
        if text:
            caps[int(ann["image_id"])].append(text)
    return caps


def sample_pairs(indexed: set[int], caps: dict[int, list[str]], n: int, seed: int):
    rng = random.Random(seed)
    pool = [i for i in sorted(indexed) if i in caps]
    if n > len(pool):
        n = len(pool)
    picked = rng.sample(pool, n)
    return [(caps[i][0], i) for i in picked]


def translate_batch(client, pairs) -> list[str]:
    """把一批英文描述改写成简短中文检索式，保持顺序。"""
    numbered = "\n".join(f"{i + 1}. {cap}" for i, (cap, _) in enumerate(pairs))
    prompt = (
        "把下面每条英文图片描述改写成一句简短的中文检索式。\n"
        "要求：不超过 15 个字；保留主体物体、场景与关键属性；不要编号、不要解释、不要标点装饰。\n"
        "只输出一个 JSON 字符串数组，长度与输入一致，顺序一致。\n\n"
        f"{numbered}"
    )
    raw = client.chat(
        [{"role": "user", "content": prompt}], json_mode=True, max_tokens=1200
    ).strip()

    try:
        start, end = raw.find("["), raw.rfind("]")
        if start >= 0 and end > start:
            arr = json.loads(raw[start : end + 1])
            out = [str(x).strip() for x in arr]
            if len(out) == len(pairs):
                return out
    except Exception:  # noqa: BLE001 - 解析失败走下面的逐行兜底
        pass

    lines = [ln.strip(" 0123456789.、)）\"'") for ln in raw.splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln]
    if len(lines) < len(pairs):
        raise RuntimeError(f"翻译结果条数不足：需要 {len(pairs)} 条，拿到 {len(lines)} 条")
    return lines[: len(pairs)]


def build_queries(captions_path: Path, manifest: Path, n: int, seed: int, batch: int) -> list[dict]:
    from findgo.retriever.config import Config
    from findgo.retriever.llm.client import LLMClient

    indexed = load_indexed_ids(manifest)
    if not indexed:
        raise RuntimeError(f"manifest 里没有可用图片：{manifest}")
    caps = load_captions(captions_path)
    pairs = sample_pairs(indexed, caps, n, seed)
    if not pairs:
        raise RuntimeError("manifest 与 captions 的 image_id 对不上，一条查询都构造不出来")

    client = LLMClient(Config.from_yaml())
    items: list[dict] = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        zh = translate_batch(client, chunk)
        for (cap, image_id), query in zip(chunk, zh):
            items.append({"query": query, "image_id": image_id, "caption": cap})
        print(f"  翻译 {min(start + batch, len(pairs))}/{len(pairs)}", flush=True)
        QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        QUERY_CACHE.write_text(
            json.dumps({"seed": seed, "n": len(items), "items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return items


# ── 评估 ────────────────────────────────────────────────────────────────────


def run_eval(searcher, items: list[dict], top_k: int, recall_k: int, use_rerank: bool) -> dict:
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0
    times: list[float] = []
    missed = 0

    for i, item in enumerate(items, 1):
        # --from-store 构造的条目只有 filename，COCO 标注构造的条目只有 image_id
        target = item.get("filename") or f"{int(item['image_id']):012d}.jpg"
        resp = searcher.search_text(
            item["query"], top_k=top_k, recall_k=recall_k, use_rerank=use_rerank
        )
        times.append(resp.elapsed)
        rank = None
        for r in resp.results:
            if getattr(r, "filename", "") == target:
                rank = getattr(r, "rank", None)
                break
        if rank is None:
            missed += 1
            continue
        for k in hits:
            if rank <= k:
                hits[k] += 1
        rr += 1.0 / rank
        if i % 10 == 0:
            print(f"    {i}/{len(items)}", flush=True)

    n = len(items)
    return {
        "n": n,
        "recall@1": hits[1] / n,
        "recall@5": hits[5] / n,
        "recall@10": hits[10] / n,
        "mrr": rr / n,
        "median_s": statistics.median(times) if times else float("nan"),
        "missed": missed,
    }


def _row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['recall@1']:.4f} | {m['recall@5']:.4f} | "
        f"{m['recall@10']:.4f} | {m['mrr']:.4f} | {m['median_s']:.2f} s |"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="6.4 中文查询检索效果实验")
    ap.add_argument("--captions", default=None, help="COCO captions_val2017.json 路径")
    ap.add_argument(
        "--from-store",
        action="store_true",
        help="从向量库已有的 caption 字段构造查询集，不需要 --captions",
    )
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=10, help="每次送 LLM 翻译的条数")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--recall-k", type=int, default=50)
    ap.add_argument("--rerank-n", type=int, default=40, help="做重排的查询条数")
    ap.add_argument("--reuse", action="store_true", help="复用已缓存的 queries_zh.json")
    ap.add_argument("--out", default="outputs/eval_zh_6_4.md")
    args = ap.parse_args()

    pipe = None
    if args.reuse:
        if not QUERY_CACHE.exists():
            print(f"没有缓存：{QUERY_CACHE}，先不带 --reuse 跑一次")
            return 1
        items = json.loads(QUERY_CACHE.read_text(encoding="utf-8"))["items"]
    elif args.from_store:
        from findgo.retriever.config import Config
        from findgo.retriever.pipeline import Pipeline

        pipe = Pipeline(Config.from_yaml())
        items = build_queries_from_store(pipe, args.n, args.seed, args.batch)
    else:
        if not args.captions:
            print("需要 --captions 指向 COCO captions_val2017.json；已有缓存则加 --reuse")
            return 1
        cap_path = Path(args.captions)
        if not cap_path.exists():
            print(f"标注文件不存在：{cap_path}")
            return 1
        items = build_queries(cap_path, Path(args.manifest), args.n, args.seed, args.batch)

    if not items:
        print("查询集为空")
        return 1

    from findgo.retriever.config import Config
    from findgo.retriever.pipeline import Pipeline

    if pipe is None:
        pipe = Pipeline(Config.from_yaml())
    try:
        count = int(pipe.store.count())
    except Exception:  # noqa: BLE001
        count = -1
    if count == 0:
        print("向量库为空，先建索引。")
        pipe.close()
        return 1

    device = str(getattr(pipe.encoder, "device", "unknown"))

    print(f"索引 {count} 条，查询 {len(items)} 条，开始纯向量评估")
    vec_all = run_eval(pipe.searcher, items, args.top_k, args.recall_k, use_rerank=False)

    subset = items[: max(1, min(args.rerank_n, len(items)))]
    print(f"子集 {len(subset)} 条：纯向量")
    vec_sub = run_eval(pipe.searcher, subset, args.top_k, args.recall_k, use_rerank=False)
    print("子集：向量 + 重排")
    rer_sub = run_eval(pipe.searcher, subset, args.top_k, args.recall_k, use_rerank=True)
    pipe.close()

    lines = [
        "# 6.4 检索效果实测（中文查询）",
        "",
        f"- 索引条数：{count}",
        f"- 查询条数：全量 {vec_all['n']}，重排子集 {rer_sub['n']}",
        f"- 计算设备：{device}",
        f"- 参数：top_k={args.top_k}，recall_k={args.recall_k}（代码内部会把 recall_k 抬到至少 top_k×2）",
        "",
        "## 表 6-2 全量纯向量",
        "",
        "| 策略 | Recall@1 | Recall@5 | Recall@10 | MRR | 单条中位耗时 |",
        "| --- | --- | --- | --- | --- | --- |",
        _row("纯向量", vec_all),
        "",
        "## 表 6-3 纯向量 vs 向量+重排（同一子集）",
        "",
        "| 策略 | Recall@1 | Recall@5 | Recall@10 | MRR | 单条中位耗时 |",
        "| --- | --- | --- | --- | --- | --- |",
        _row("纯向量", vec_sub),
        _row("向量 + LLM 重排", rer_sub),
        "",
        "## 表 6-4 重排增益（百分点）",
        "",
        "| 指标 | 纯向量 | 加重排 | 差值 |",
        "| --- | --- | --- | --- |",
    ]
    for key in ("recall@1", "recall@5", "recall@10", "mrr"):
        lines.append(
            f"| {key} | {vec_sub[key]:.4f} | {rer_sub[key]:.4f} | "
            f"{(rer_sub[key] - vec_sub[key]) * 100:+.2f} |"
        )
    lines += [
        "",
        "全量组的漏检数（目标图不在返回结果里）：" + str(vec_all["missed"]),
        "",
        "写论文时注意：重排只对前若干条查询跑，对比必须用同一子集；"
        "Recall@10 的上限受召回候选数限制（recall_k 会被抬到至少 top_k×2），"
        "若出现 Recall@10 低于 Recall@5，说明候选集太小，把 --recall-k 调大重跑。",
        "",
    ]
    table = "\n".join(lines)
    print("\n" + table)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")

    # 结构化副本：给 scripts/fill_6_4.py 填论文表格用，免得再从 Markdown 里抠数
    payload = {
        "indexed": count,
        "device": device,
        "top_k": args.top_k,
        "recall_k": args.recall_k,
        "n_all": vec_all["n"],
        "n_sub": rer_sub["n"],
        "vec_all": vec_all,
        "vec_sub": vec_sub,
        "rer_sub": rer_sub,
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] 已写入 {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
