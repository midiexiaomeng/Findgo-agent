"""准备一个真实图片测试库。

默认从 COCO val2017 里筛出满足分辨率要求的图片，抽样 N 张，生成
manifest 和若干条中文测试查询。

    python scripts/prepare_test_dataset.py --download
    python scripts/prepare_test_dataset.py --zip D:\\val2017.zip
    python scripts/prepare_test_dataset.py --from-dir D:\\Photos --limit 2000

产物：
    datasets/coco_2000/images/*.jpg
    datasets/coco_2000/manifest.json
    datasets/coco_2000/sample_queries.txt
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# 这些查询覆盖 COCO 里的常见场景，可以直接粘到搜索框里试
SAMPLE_QUERIES = [
    "一只猫",
    "一只狗在草地上",
    "几个人在餐厅吃饭",
    "一辆红色的公交车",
    "海边的沙滩",
    "雪地里的滑雪者",
    "桌子上的披萨",
    "一个人在骑自行车",
    "城市街道的夜景",
    "一只鸟站在树枝上",
    "厨房里的冰箱",
    "棒球比赛",
    "一杯咖啡",
    "飞机停在机场",
    "卧室里的床",
    "一群人走在街上",
    "一匹马",
    "生日蛋糕",
    "笔记本电脑和键盘",
    "夕阳下的海滩",
]


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[跳过] 已存在 {dest.name} ({human_size(dest.stat().st_size)})")
        return dest

    print(f"[下载] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 / total
                    print(
                        f"\r  {human_size(done)} / {human_size(total)}  {pct:5.1f}%",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  {human_size(done)}", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def extract(zip_path: Path, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"[跳过] 已解压 {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[解压] {zip_path.name} -> {dest}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def scan_images(folder: Path, min_side: int) -> list[tuple[Path, int, int]]:
    from PIL import Image

    found: list[tuple[Path, int, int]] = []
    files = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    print(f"[扫描] {len(files)} 个文件，检查分辨率 ...")
    for i, p in enumerate(files, 1):
        try:
            with Image.open(p) as im:
                w, h = im.size
            if min(w, h) >= min_side:
                found.append((p, w, h))
        except Exception:
            continue
        if i % 500 == 0:
            print(f"  {i}/{len(files)}")
    print(f"[筛选] 满足最短边 >= {min_side}px 的有 {len(found)} 张")
    return found


def load_captions(ann_zip: Path) -> dict[int, str]:
    if not ann_zip.exists():
        return {}
    try:
        with zipfile.ZipFile(ann_zip) as z:
            name = next(
                (n for n in z.namelist() if n.endswith("captions_val2017.json")), None
            )
            if not name:
                return {}
            with z.open(name) as f:
                data = json.load(f)
        captions: dict[int, str] = {}
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            if image_id not in captions:
                captions[image_id] = str(ann.get("caption", "")).strip()
        return captions
    except Exception as e:
        print(f"[警告] 读取标注失败: {e}")
        return {}


def build(
    source_dir: Path,
    out_dir: Path,
    limit: int,
    min_side: int,
    seed: int,
    captions: dict[int, str] | None = None,
) -> None:
    images = scan_images(source_dir, min_side)
    if not images:
        raise SystemExit("没有满足条件的图片，换个目录或降低 --min-side")

    rng = random.Random(seed)
    if len(images) > limit:
        picked = rng.sample(images, limit)
    else:
        picked = images
    picked.sort(key=lambda x: x[0].name)

    image_dir = out_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_bytes = 0
    for p, w, h in picked:
        target = image_dir / p.name
        shutil.copy2(p, target)
        size = target.stat().st_size
        total_bytes += size
        image_id = None
        if p.stem.isdigit():
            image_id = int(p.stem)
        records.append(
            {
                "file": p.name,
                "width": w,
                "height": h,
                "size_bytes": size,
                "caption": (captions or {}).get(image_id, "") if image_id else "",
            }
        )

    manifest = {
        "source": str(source_dir),
        "count": len(records),
        "min_side": min_side,
        "seed": seed,
        "total_bytes": total_bytes,
        "images": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "sample_queries.txt").write_text(
        "\n".join(SAMPLE_QUERIES), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"输出目录 : {out_dir}")
    print(f"图片数量 : {len(records)}")
    print(f"总大小   : {human_size(total_bytes)}")
    print(f"平均尺寸 : {sum(r['width'] for r in records) // len(records)} x "
          f"{sum(r['height'] for r in records) // len(records)}")
    print(f"带标注   : {sum(1 for r in records if r['caption'])} 张")
    print()
    print("下一步：")
    print(f'  python cli.py index "{out_dir / "images"}"')
    print("  python app.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="准备真实图片测试库")
    parser.add_argument("--download", action="store_true", help="自动下载 COCO val2017")
    parser.add_argument("--zip", default=None, help="本地已有的 val2017.zip")
    parser.add_argument("--from-dir", default=None, help="使用自己的图片目录")
    parser.add_argument("--out", default=str(ROOT / "datasets" / "coco_2000"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--min-side", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-work", action="store_true", help="保留解压的原始文件")
    parser.add_argument("--no-captions", action="store_true", help="不下载标注（省 241MB）")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    captions: dict[int, str] = {}

    if args.from_dir:
        source = Path(args.from_dir).expanduser().resolve()
        if not source.exists():
            print(f"目录不存在: {source}")
            return 1
        build(source, out_dir, args.limit, args.min_side, args.seed)

    elif args.zip or args.download:
        work = out_dir.parent / "_coco_work"
        work.mkdir(parents=True, exist_ok=True)

        if args.zip:
            zip_path = Path(args.zip).expanduser().resolve()
            if not zip_path.exists():
                print(f"zip 不存在: {zip_path}")
                return 1
        else:
            zip_path = download(COCO_VAL_URL, work / "val2017.zip")

        ann_zip = work / "annotations_trainval2017.zip"
        if not args.no_captions:
            if not ann_zip.exists():
                try:
                    download(COCO_ANN_URL, ann_zip)
                except Exception as e:
                    print(f"[警告] 标注下载失败，将不生成 caption: {e}")
            captions = load_captions(ann_zip)

        extract(zip_path, work)
        source = work / "val2017"
        if not source.exists():
            candidates = [p for p in work.rglob("*.jpg")]
            if not candidates:
                print("解压后没有找到 jpg，请检查 zip 结构")
                return 1
            source = candidates[0].parent

        build(source, out_dir, args.limit, args.min_side, args.seed, captions)

        if not args.keep_work:
            print(f"[清理] 删除中间文件 {work}")
            shutil.rmtree(work, ignore_errors=True)
    else:
        parser.print_help()
        print("\n至少指定一个来源：--download / --zip / --from-dir")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
