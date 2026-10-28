"""pytest 自动加载：把 src/ 和项目根加入 sys.path，并给 Qt 设置无头模式。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# CI 与本地无窗口环境下，Qt 需要有平台插件才能初始化
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
