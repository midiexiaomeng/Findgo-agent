"""Findgo：基于 ReAct 的多模态图像检索 Agent。"""

from pathlib import Path


def _read_version() -> str:
    """优先读项目根的 VERSION；安装成包后回退到硬编码值。"""
    try:
        root = Path(__file__).resolve().parents[2]
        return (root / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        return "0.1.0"


__version__ = _read_version()
