"""把 6.4 的实测数一次性填进论文合并稿。

配合 scripts/eval_zh_6_4.py 使用：先跑评估（它现在会同时写出
outputs/eval_zh_6_4.md 和 outputs/eval_zh_6_4.json），再跑本脚本，
把下列 27 处占位换成实测值：

    N 的 6 处            中文摘要、Abstract、6.4 正文、表 6-2 表题、6.5 小结、7.2 不足
    摘要里的 Recall@1     中文摘要、Abstract 各一处「由 ⚠ 提升至 ⚠」
    表 6-2 的 6 格      Recall@1/5/10 × 纯向量、向量+重排
    表 6-3 的 4 格      MRR@10 与单条中位耗时 × 两种策略
    表 6-4 的 6 格      绝对提升（百分点）与相对提升

并把 6.4 的分析段成品替换掉那句「本段分析待三表填实后补写」。

口径：三张表都用重排子集（同一查询集），N 取子集的条数；全量组的
读数只打印出来备查，不进表格。

默认只预览，不改文件：

    python scripts\\fill_6_4.py                 # 预览
    python scripts\\fill_6_4.py --apply         # 落盘（自动备份）

若三张表的增益全为 0，脚本会拒绝填写并提示重排后端没生效，因为那说明
6.4 跑出来的仍是纯向量口径，交上去等于自己打自己。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "outputs" / "eval_zh_6_4.json"
DEFAULT_THESIS = Path(r"D:\计算机组成原理文件\毕业论文材料\毕业论文合并稿_v1.md")


class PatchError(RuntimeError):
    """锚点没对上就整体放弃，绝不半途改稿。"""


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def pp(delta: float) -> str:
    """绝对提升，单位百分点。"""
    return f"{delta * 100:+.1f}".replace("+0.0", "0.0")


def rel(delta: float, base: float) -> str:
    if base <= 0:
        return "—"
    return f"{delta / base * 100:+.1f}%".replace("+0.0%", "0.0%")


def sub_once(text: str, pattern: str, repl: str, label: str) -> str:
    hits = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(hits) != 1:
        raise PatchError(f"{label}：命中 {len(hits)} 处，期望 1 处（{pattern}）")
    m = hits[0]
    return text[: m.start()] + repl + text[m.end() :]


def build_analysis(vec: dict, rer: dict) -> str:
    """按实测关系拼 6.4 的分析段，四个分支各说各的。"""
    a1, b1 = pct(vec["recall@1"]), pct(rer["recall@1"])
    a5, b5 = pct(vec["recall@5"]), pct(rer["recall@5"])
    a10, b10 = pct(vec["recall@10"]), pct(rer["recall@10"])
    am, bm = f"{vec['mrr']:.3f}", f"{rer['mrr']:.3f}"
    tv, tr = vec["median_s"], rer["median_s"]
    d1 = rer["recall@1"] - vec["recall@1"]
    d10 = rer["recall@10"] - vec["recall@10"]
    dm = rer["mrr"] - vec["mrr"]

    parts = [
        f"从表 6-2 可见，引入语义重排后 Recall@1 由 {a1} 变为 {b1}"
        f"（绝对提升 {pp(d1)} 个百分点，相对提升 {rel(d1, vec['recall@1'])}），"
        f"Recall@5 由 {a5} 变为 {b5}。"
    ]

    if d10 > 0.001:
        parts.append(
            f"Recall@10 由 {a10} 升至 {b10}，说明重排在扩大的候选集上同样找回了更多目标图。"
        )
    elif abs(d10) <= 0.001:
        parts.append(
            f"Recall@10 保持在 {b10}，未随重排变化，说明召回候选集本已覆盖目标图，"
            "重排的作用集中在排序前端而非提高上限。"
        )
    else:
        parts.append(
            f"Recall@10 由 {a10} 小幅回落到 {b10}，原因是重排只在截断后的候选内调整顺序，"
            "极少数原本靠后命中的目标图被移出了前 10 位。"
        )

    if dm > 0.0005:
        parts.append(
            f"表 6-3 的 MRR@10 由 {am} 升至 {bm}，与 Recall@1 的走势一致，"
            "表明重排确实把目标图整体前移，作用不限于个例。"
        )
    else:
        parts.append(
            f"表 6-3 的 MRR@10 由 {am} 变为 {bm}，排序质量的改善幅度有限，"
            "这与向量召回本身已把多数目标图排在前列相符。"
        )

    if tr > tv:
        parts.append(
            f"代价是单条中位耗时由 {tv:.2f} 秒增至 {tr:.2f} 秒"
            f"（约 {(tr / tv - 1) * 100:.0f}%），增量来自对每条查询的候选逐个调用语言模型打分，"
            "按一次交互的等待时间衡量仍在可接受范围内。"
        )
    else:
        parts.append(
            f"单条中位耗时由 {tv:.2f} 秒变为 {tr:.2f} 秒，未出现可感知的额外开销，"
            "这与重排批次较小、且模型常驻显存有关。"
        )

    parts.append(
        "需要再次说明的是，本节查询由库内描述改写而来，与重排所读的候选描述同源，"
        "上述增益应视为上限估计，真实用户查询下的收益会低于表中数值。"
    )
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="把 6.4 实测数填进论文合并稿")
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="eval_zh_6_4.py 写出的 json")
    ap.add_argument("--thesis", default=str(DEFAULT_THESIS), help="论文合并稿 md")
    ap.add_argument("--apply", action="store_true", help="真正写回文件（默认只预览）")
    args = ap.parse_args()

    data_path = Path(args.json)
    thesis_path = Path(args.thesis)
    if not data_path.exists():
        print(f"没有实测数据：{data_path}\n先跑 python scripts\\eval_zh_6_4.py --from-store --n 200")
        return 1
    if not thesis_path.exists():
        print(f"找不到论文稿：{thesis_path}")
        return 1

    data = json.loads(data_path.read_text(encoding="utf-8"))
    vec_sub, rer_sub = data["vec_sub"], data["rer_sub"]
    n = int(rer_sub["n"])

    deltas = {
        "Recall@1": rer_sub["recall@1"] - vec_sub["recall@1"],
        "Recall@10": rer_sub["recall@10"] - vec_sub["recall@10"],
        "MRR@10": rer_sub["mrr"] - vec_sub["mrr"],
    }
    if all(abs(v) < 1e-9 for v in deltas.values()):
        print(
            "三张表的增益全为 0：两种策略读数完全相同，说明重排后端没生效"
            "（查 config.yaml 的 rerank.backend 与 5.4 的描述），"
            "这组数不能填进论文，先修后端再重跑。"
        )
        return 2

    raw = thesis_path.read_text(encoding="utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n") if crlf else raw
    before = text.count("⚠")

    try:
        # N 的六处
        text = sub_once(text, r"由图像描述改写的 ⚠ 条中文查询",
                        f"由图像描述改写的 {n} 条中文查询", "N-中文摘要")
        text = sub_once(text, r"On ⚠ Chinese queries",
                        f"On {n} Chinese queries", "N-Abstract")
        text = sub_once(text, r"同一组 N=⚠ 条查询",
                        f"同一组 N={n} 条查询", "N-6.4正文")
        text = sub_once(text, r"（同一查询集，N=⚠）",
                        f"（同一查询集，N={n}）", "N-表6-2表题")
        text = sub_once(text, r"检索效果层面在 ⚠ 条中文查询上",
                        f"检索效果层面在 {n} 条中文查询上", "N-6.5小结")
        text = sub_once(text, r"单一数据集的 ⚠ 条查询上",
                        f"单一数据集的 {n} 条查询上", "N-7.2不足")

        # 摘要里的 Recall@1 起止值
        text = sub_once(
            text, r"Recall@1 由 ⚠ 提升至 ⚠",
            f"Recall@1 由 {pct(vec_sub['recall@1'])} 提升至 {pct(rer_sub['recall@1'])}",
            "摘要-中文Recall@1")
        text = sub_once(
            text, r"from ⚠ to ⚠",
            f"from {pct(vec_sub['recall@1'])} to {pct(rer_sub['recall@1'])}",
            "摘要-AbstractRecall@1")

        # 表 6-2
        text = sub_once(
            text, r"^\| 纯向量检索 \| ⚠ \| ⚠ \| ⚠ \|$",
            f"| 纯向量检索 | {pct(vec_sub['recall@1'])} | {pct(vec_sub['recall@5'])} "
            f"| {pct(vec_sub['recall@10'])} |", "表6-2-纯向量")
        text = sub_once(
            text, r"^\| 向量 \+ 重排 \| ⚠ \| ⚠ \| ⚠ \|$",
            f"| 向量 + 重排 | {pct(rer_sub['recall@1'])} | {pct(rer_sub['recall@5'])} "
            f"| {pct(rer_sub['recall@10'])} |", "表6-2-加重排")

        # 表 6-3
        text = sub_once(
            text, r"^\| 纯向量检索 \| ⚠ \| ⚠ \|$",
            f"| 纯向量检索 | {vec_sub['mrr']:.3f} | {vec_sub['median_s']:.2f} |", "表6-3-纯向量")
        text = sub_once(
            text, r"^\| 向量 \+ 重排 \| ⚠ \| ⚠ \|$",
            f"| 向量 + 重排 | {rer_sub['mrr']:.3f} | {rer_sub['median_s']:.2f} |", "表6-3-加重排")

        # 表 6-4
        for key, base in (
            ("Recall@1", vec_sub["recall@1"]),
            ("Recall@10", vec_sub["recall@10"]),
            ("MRR@10", vec_sub["mrr"]),
        ):
            d = deltas[key]
            text = sub_once(
                text, rf"^\| {key} \| ⚠ \| ⚠ \|$",
                f"| {key} | {pp(d)} | {rel(d, base)} |", f"表6-4-{key}")

        # 6.4 分析段
        text = sub_once(
            text, r"^⚠ 本段分析与结论待三表填实后补写.*$",
            build_analysis(vec_sub, rer_sub), "6.4分析段")
    except PatchError as exc:
        print(f"[中止] {exc}\n文件未改动。稿子可能又改过，把锚点对准了再跑。")
        return 3

    after = text.count("⚠")
    print(f"⚠ 由 {before} 处减到 {after} 处（本轮填 {before - after} 处）")
    for line_no, line in enumerate(text.split("\n"), 1):
        if "⚠" in line and not line.lstrip().startswith(">"):
            print(f"  剩 L{line_no}: {line.strip()[:90]}")

    print("\n全量组（不进表格，备查）："
          f"n={data['n_all']} Recall@1={pct(data['vec_all']['recall@1'])} "
          f"MRR={data['vec_all']['mrr']:.3f} 漏检={data['vec_all']['missed']}")

    if not args.apply:
        print("\n预览完毕，没有写文件。确认无误后加 --apply。")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = thesis_path.with_suffix(f".bak-{stamp}.md")
    shutil.copy2(thesis_path, backup)
    out = text.replace("\n", "\r\n") if crlf else text
    thesis_path.write_text(out, encoding="utf-8")
    print(f"\n已写回 {thesis_path}\n备份 {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
