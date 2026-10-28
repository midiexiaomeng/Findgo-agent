"""运行时工具：日志、设备、精度、路径解析。

放在这里的都是"与业务无关但每个模块都要用"的东西，避免各模块重复造轮子。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # retriever/runtime.py → 项目根


# ── 日志 ──────────────────────────────────────────────────────────────────────


def setup_logging(level: str = "INFO") -> None:
    """统一日志格式。重复调用是安全的。"""
    root = logging.getLogger()
    if getattr(root, "_findgo_configured", False):
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root._findgo_configured = True  # type: ignore[attr-defined]

    # 第三方库的噪音压掉
    for noisy in ("httpx", "urllib3", "chromadb.telemetry", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── 设备与精度 ────────────────────────────────────────────────────────────────


def pick_device(preference: str = "auto") -> str:
    """把 auto 解析成实际可用设备。"""
    import torch

    if preference and preference != "auto":
        if preference.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("配置指定了 %s，但当前 CUDA 不可用，回退到 CPU", preference)
            return "cpu"
        return preference

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, prefer_fp16: bool = True):
    """GPU 上用 fp16 省显存；CPU/MPS 上用 fp32 求稳。"""
    import torch

    if prefer_fp16 and device.startswith("cuda"):
        return torch.float16
    return torch.float32


def describe_device(device: str) -> str:
    """给界面用的一行设备描述。"""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                idx = 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return f"{name} ({total:.1f} GB)"
    if device == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


# ── 路径 ──────────────────────────────────────────────────────────────────────


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """相对路径一律相对项目根目录解析，避免双击启动时 cwd 不对。"""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p


def resolve_model_source(
    configured: str,
    models_dir: str | Path,
    fallback: str,
) -> str:
    """模型解析：本地目录优先，其次 HF 仓库 ID。

    配置里写短名（如 chinese-clip-vit-base-patch16）时：
      1. models_dir/<短名> 存在且非空 -> 用本地
      2. configured 本身是存在的路径 -> 用该路径
      3. 否则回退到 HuggingFace 仓库 ID（首次运行会联网下载）
    """
    models_dir = Path(models_dir)

    # 1) 配置值当成路径
    candidate = Path(configured).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if not candidate.is_absolute():
        local = resolve_path(candidate)
        if local.exists():
            return str(local)

    # 2) models_dir 下的短名
    short = candidate.name if candidate.name else configured
    local_dir = models_dir / short
    if local_dir.exists() and any(local_dir.iterdir()):
        return str(local_dir)

    # 3) 回退 HF
    logger.warning(
        "未在本地找到模型 %s，将使用 HuggingFace 源: %s（首次运行需联网下载）",
        configured,
        fallback,
    )
    return fallback


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def human_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def set_windows_safe_env() -> None:
    """Windows 上 torch 与 chromadb 的 OpenMP 运行库偶尔打架，提前规避。"""
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
