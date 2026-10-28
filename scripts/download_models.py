"""一键下载模型。

    python scripts/download_models.py                 # CLIP + BLIP
    python scripts/download_models.py --with-llm      # 再加 Qwen2.5-7B（走 Ollama）
    python scripts/download_models.py --check         # 只检查，不下载
    python scripts/download_models.py --mirror https://hf-mirror.com

模型不会打进发布包：CLIP 约 400MB，BLIP 约 1GB，7B 模型约 4.7GB（Q4）。
下载是幂等的、可断点续传的，重复执行只会补齐缺的文件。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = {
    "clip": {
        "repo": "OFA-Sys/chinese-clip-vit-base-patch16",
        "dir": "chinese-clip-vit-base-patch16",
        "label": "Chinese-CLIP ViT-B/16（图文嵌入，约 400MB）",
    },
    "blip": {
        "repo": "Salesforce/blip-image-captioning-base",
        "dir": "blip-image-captioning-base",
        "label": "BLIP caption base（图像描述，约 1GB）",
    },
}

DONE_MARKER = ".download_complete"


# ── 工具 ──────────────────────────────────────────────────────────────────────


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def is_complete(path: Path) -> bool:
    return (path / DONE_MARKER).exists() and any(
        p.suffix in (".bin", ".safetensors") for p in path.rglob("*")
    )


def find_ollama() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


# ── 下载 ──────────────────────────────────────────────────────────────────────


def download_hf(key: str, models_dir: Path, mirror: str | None) -> bool:
    info = MODELS[key]
    target = models_dir / info["dir"]

    if is_complete(target):
        print(f"[跳过] {info['dir']} 已完整 ({human_size(dir_size(target))})")
        return True

    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"使用镜像: {mirror}")

    print(f"[下载] {info['label']}")
    print(f"       {info['repo']} -> {target}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ! 缺少 huggingface_hub，请先 pip install -r requirements.txt")
        return False

    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        snapshot_download(
            repo_id=info["repo"],
            local_dir=str(target),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pt", ".gitattributes", "*.md"],
            max_workers=4,
        )
    except Exception as e:
        print(f"  ! 下载失败: {e}")
        print("    可以换镜像重试：--mirror https://hf-mirror.com")
        return False

    # 优先保留 safetensors，删掉重复的 bin
    safetensors = list(target.glob("*.safetensors"))
    if safetensors:
        for bin_file in target.glob("pytorch_model*.bin"):
            try:
                bin_file.unlink()
            except OSError:
                pass

    (target / DONE_MARKER).write_text(
        json.dumps(
            {
                "repo": info["repo"],
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": dir_size(target),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  完成，用时 {time.time() - started:.0f}s，{human_size(dir_size(target))}")
    return True


def download_llm(model: str, auto_install: bool) -> bool:
    exe = find_ollama()
    if not exe:
        print("[LLM] 未检测到 Ollama，跳过语言模型（CLIP/BLIP 下载不受影响）")
        print("      Windows 安装: winget install Ollama.Ollama")
        print("      然后重跑: python scripts/download_models.py --with-llm")
        print("      没有 LLM 也能检索，只是自动退化为纯向量排序。")
        return True

    print(f"[LLM] 使用 Ollama 拉取 {model}（约 4.7GB，首次较慢）")
    try:
        result = subprocess.run([exe, "pull", model], check=False)
        if result.returncode != 0:
            print(f"  ! ollama pull 失败，退出码 {result.returncode}")
            return False
    except Exception as e:
        print(f"  ! 调用 Ollama 失败: {e}")
        return False

    print(f"  完成。确认服务已启动：{exe} serve")
    return True


# ── 检查 ──────────────────────────────────────────────────────────────────────


def check(models_dir: Path, llm_model: str) -> int:
    print(f"模型目录: {models_dir}")
    print("-" * 60)
    missing = 0
    for key, info in MODELS.items():
        target = models_dir / info["dir"]
        if is_complete(target):
            print(f"[OK]   {info['dir']:<42} {human_size(dir_size(target))}")
        elif target.exists():
            print(f"[部分] {info['dir']:<42} 有目录但未完成")
            missing += 1
        else:
            print(f"[缺失] {info['dir']:<42} {info['repo']}")
            missing += 1

    exe = find_ollama()
    print("-" * 60)
    if not exe:
        print("[LLM] 未安装 Ollama（重排会退化为纯向量排序）")
    else:
        try:
            out = subprocess.run(
                [exe, "list"], capture_output=True, text=True, timeout=15
            )
            has_model = llm_model.split(":")[0] in out.stdout
            print(f"[LLM] Ollama 已安装; 模型 {llm_model}: {'已就绪' if has_model else '未下载'}")
            if not has_model:
                print(f"      运行: ollama pull {llm_model}")
        except Exception as e:
            print(f"[LLM] 无法查询 Ollama: {e}")

    return 0 if missing == 0 else 1


# ── 入口 ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 findgo 所需模型")
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument("--mirror", default="https://hf-mirror.com",
                        help="HuggingFace 镜像，传空字符串使用官方源")
    parser.add_argument("--with-llm", action="store_true", help="同时拉取 Qwen2.5-7B（Ollama）")
    parser.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--check", action="store_true", help="只检查，不下载")
    parser.add_argument("--only", choices=list(MODELS), default=None, help="只下载某一个")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        return check(models_dir, args.llm_model)

    mirror = args.mirror.strip() or None
    keys = [args.only] if args.only else list(MODELS)
    results = {k: download_hf(k, models_dir, mirror) for k in keys}

    if args.with_llm:
        results["llm"] = download_llm(args.llm_model, auto_install=False)

    print()
    print("=" * 60)
    failed = [k for k, ok in results.items() if not ok]
    if failed:
        print(f"未完成: {', '.join(failed)}")
        print("可以重复执行本脚本续传；网络不稳时加 --mirror https://hf-mirror.com")
        return 1

    print("模型准备完成。下一步：python app.py")
    print("提示: 首次检索会加载模型进显存，之后就是常驻的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
