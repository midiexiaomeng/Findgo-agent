"""环境自检：给设置面板和 Agent 工具共用的一组检查。

只做只读探测，不加载模型、不连接向量库，避免自检本身把环境搞乱。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _ok(name: str, detail: str) -> dict:
    return {"name": name, "ok": True, "detail": detail}


def _bad(name: str, detail: str) -> dict:
    return {"name": name, "ok": False, "detail": detail}


def check_python() -> dict:
    version = sys.version.split()[0]
    if sys.version_info >= (3, 9):
        return _ok("Python", version)
    return _bad("Python", f"{version}，需要 3.9 及以上")


def check_dependencies() -> list[dict]:
    # 项目把 qdrant_client / portalocker 放在 _vendor，先把路径挂上再检查
    vendor = Path(__file__).resolve().parents[2] / "_vendor"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    results = []
    modules = [
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("qdrant_client", "qdrant-client"),
    ]
    for module, label in modules:
        try:
            mod = __import__(module)
            results.append(_ok(label, getattr(mod, "__version__", "已安装")))
        except Exception as e:
            results.append(_bad(label, f"缺失（{e}）"))
    return results


def check_cuda() -> dict:
    try:
        import torch
    except Exception as e:
        return _bad("CUDA", f"torch 不可用：{e}")

    try:
        if not torch.cuda.is_available():
            return _ok("CUDA", "不可用，将使用 CPU")
        parts = []
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            parts.append(f"{prop.name} {prop.total_memory / 1024**3:.1f}GB")
        return _ok("CUDA", " | ".join(parts))
    except Exception as e:
        return _bad("CUDA", f"检测失败：{e}")


def check_models() -> list[dict]:
    from .retriever import Config
    from .retriever.runtime import resolve_model_source

    try:
        cfg = Config.from_yaml()
    except Exception as e:
        return [_bad("模型配置", f"读取失败：{e}")]

    results = []
    for label, name, fallback in [
        ("CLIP 模型", cfg.embedding.model, cfg.embedding.fallback),
        ("BLIP 模型", cfg.caption.model, cfg.caption.fallback),
    ]:
        try:
            source = resolve_model_source(name, cfg.models_dir, fallback)
            path = Path(source)
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                results.append(_ok(label, f"{path.name}（{size / 1024**2:.0f} MB）"))
            else:
                results.append(_bad(label, f"本地不存在：{source}"))
        except Exception as e:
            results.append(_bad(label, f"检测失败：{e}"))
    return results


def check_vector_store() -> dict:
    from .retriever import Config

    try:
        cfg = Config.from_yaml()
    except Exception as e:
        return _bad("向量库", f"配置读取失败：{e}")

    backend = str(cfg.vector_store.backend).lower()
    try:
        if backend.startswith("qdrant"):
            path = cfg.resolve(cfg.vector_store.qdrant.path)
            label = "Qdrant"
        else:
            path = cfg.resolve(cfg.vector_store.chromadb.path)
            label = "Chroma"

        path = Path(path)
        if not path.exists():
            return _bad("向量库", f"{label} 目录不存在：{path}")
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return _ok("向量库", f"{label} / {cfg.vector_store.collection}（{size / 1024**2:.0f} MB）")
    except Exception as e:
        return _bad("向量库", f"检测失败：{e}")


def check_llm() -> dict:
    try:
        from .model_store import ModelStore

        entry = ModelStore().get_active()
        if entry:
            return _ok("LLM", f"{entry.get('label') or entry.get('model', '')}")
    except Exception:
        pass

    if os.environ.get("DEEPSEEK_API_KEY"):
        return _ok("LLM", "环境变量 DEEPSEEK_API_KEY 已配置")
    return _bad("LLM", "未配置模型，请在「添加模型」里填写并测试")


def check_disk() -> dict:
    from .retriever import Config

    try:
        cfg = Config.from_yaml()
        root = Path(cfg.resolve(cfg.paths.data_dir))
    except Exception:
        root = Path.cwd()

    try:
        usage = shutil.disk_usage(str(root))
        free_gb = usage.free / 1024**3
        if free_gb < 5:
            return _bad("磁盘空间", f"仅剩 {free_gb:.1f} GB")
        return _ok("磁盘空间", f"可用 {free_gb:.1f} GB")
    except Exception as e:
        return _bad("磁盘空间", f"检测失败：{e}")


def run_checks() -> list[dict]:
    """返回所有检查项，按展示顺序排列。"""
    checks: list[dict] = [check_python()]
    checks.extend(check_dependencies())
    checks.append(check_cuda())
    checks.extend(check_models())
    checks.append(check_vector_store())
    checks.append(check_llm())
    checks.append(check_disk())
    return checks


def summarize(checks: list[dict]) -> str:
    total = len(checks)
    failed = sum(1 for c in checks if not c["ok"])
    if failed == 0:
        return f"全部 {total} 项通过"
    return f"{total} 项中 {failed} 项异常"
