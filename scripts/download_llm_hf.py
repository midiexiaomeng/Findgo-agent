"""下载 HuggingFace 格式的本地 LLM，供 transformers 后端使用。

不需要 Ollama，也不需要任何推理服务。模型下载到 models/ 下，
程序用 transformers 直接加载。

    python scripts/download_llm_hf.py --size 3b
    python scripts/download_llm_hf.py --size 1.5b
    python scripts/download_llm_hf.py --model Qwen/Qwen2.5-3B-Instruct --dir models/Qwen2.5-3B-Instruct
    python scripts/download_llm_hf.py --check

显存参考（fp16）：
    1.5B  约 3.1GB  适合 6GB 以上显存
    3B    约 6.2GB  适合 8GB 显存
    7B    约 15GB   8GB 显存跑不动
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIZES = {
    "0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct"),
    "3b": ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
    "7b": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
}

DONE_MARKER = ".download_complete"


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def is_complete(path: Path) -> bool:
    if not (path / DONE_MARKER).exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def download(repo: str, target: Path, mirror: str | None) -> bool:
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"mirror: {mirror}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub missing. pip install -r requirements-core.txt")
        return False

    target.mkdir(parents=True, exist_ok=True)
    print(f"repo  : {repo}")
    print(f"target: {target}")
    print("downloading ...")

    started = time.time()
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.md", "*.h5"],
            max_workers=4,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        print("retry with: --mirror https://hf-mirror.com")
        return False

    # 如果有重复的 bin 权重，删掉省空间
    for f in target.glob("pytorch_model*.bin"):
        try:
            f.unlink()
        except OSError:
            pass

    (target / DONE_MARKER).write_text(
        json.dumps(
            {
                "repo": repo,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": dir_size(target),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done: {human_size(dir_size(target))}, {time.time() - started:.0f}s")
    return True


def check(models_dir: Path) -> int:
    found = 0
    for size, (repo, name) in SIZES.items():
        path = models_dir / name
        if is_complete(path):
            print(f"[OK]   {name:<24} {human_size(dir_size(path))}")
            found += 1
        elif path.exists():
            print(f"[部分] {name:<24} 目录存在但未完成")
        else:
            print(f"[缺失] {name:<24} {repo}")
    return 0 if found else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="下载本地 LLM（HuggingFace 格式）")
    parser.add_argument("--size", choices=list(SIZES), default="3b")
    parser.add_argument("--model", default=None, help="自定义 HF 仓库 ID")
    parser.add_argument("--dir", default=None, help="目标目录")
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()

    if args.check:
        return check(models_dir)

    if args.model:
        repo = args.model
        name = args.dir or Path(repo).name
        target = Path(name).expanduser()
        if not target.is_absolute():
            target = models_dir / name
    else:
        repo, name = SIZES[args.size]
        target = Path(args.dir).expanduser() if args.dir else models_dir / name
        if not target.is_absolute():
            target = models_dir / target

    if is_complete(target):
        print(f"already complete: {target} ({human_size(dir_size(target))})")
        return 0

    mirror = args.mirror.strip() or None
    ok = download(repo, target, mirror)
    if not ok:
        return 1

    print()
    print("=" * 60)
    print("下一步：把 config.yaml 改成")
    print("  rerank:")
    print("    backend: transformers")
    print(f"    transformers:")
    print(f"      model_path: {target}")
    print()
    print("然后重启程序。首次搜索会加载模型，约 30-60 秒。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
