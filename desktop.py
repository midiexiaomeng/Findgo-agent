"""桌面窗口启动器。

不改变任何后端逻辑，只是把同一个界面装进一个独立窗口：

    python desktop.py                 自动选：pywebview > Edge 独立窗口 > 浏览器
    python desktop.py --mode webview  Electron 风格的原生窗口（需 pip install pywebview）
    python desktop.py --mode edge     Edge 的 --app 模式，无地址栏、无标签页
    python desktop.py --mode browser  普通浏览器
    python desktop.py --check         只检查环境，不开窗口

双击 ``启动应用.bat`` 会以无控制台方式运行本文件。
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WINDOW_TITLE = "多模态图像检索"
WINDOW_SIZE = (1440, 920)
MIN_SIZE = (1080, 700)


# ── 环境 ──────────────────────────────────────────────────────────────────────


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_port(host: str, port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(host, port):
            return True
        time.sleep(0.3)
    return False


def find_edge() -> Path | None:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for path in candidates:
        try:
            if path.exists():
                return path
        except OSError:
            continue
    return None


def has_webview() -> bool:
    try:
        import webview  # noqa: F401

        return True
    except Exception:
        return False


def redirect_output_when_no_console() -> None:
    """pythonw 启动时没有控制台，把输出写到日志，否则出错完全看不到。"""
    if "pythonw" not in Path(sys.executable).name.lower():
        return
    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = open(log_dir / "desktop.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = stream
        sys.stderr = stream
    except Exception:
        pass


# ── 服务 ──────────────────────────────────────────────────────────────────────


def start_server(host: str, port: int):
    from imgsearch import Config
    from imgsearch.runtime import setup_logging
    from webui.app import CSS, build_app

    cfg = Config.from_yaml()
    setup_logging(cfg.app.log_level)

    import gradio as gr

    app = build_app()
    app.queue(default_concurrency_limit=1)

    kwargs = {
        "server_name": host,
        "server_port": port,
        "inbrowser": False,
        "prevent_thread_lock": True,
        "quiet": True,
        "show_error": True,
    }
    try:
        major = int(gr.__version__.split(".")[0])
    except Exception:
        major = 4
    if major >= 6:
        # Gradio 6 把 theme/css 从 Blocks 挪到了 launch
        kwargs["theme"] = gr.themes.Soft()
        kwargs["css"] = CSS

    app.launch(**kwargs)
    return app


# ── 窗口 ──────────────────────────────────────────────────────────────────────


def run_webview(url: str, title: str = WINDOW_TITLE) -> bool:
    try:
        import webview
    except ImportError:
        return False

    webview.create_window(
        title,
        url,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=MIN_SIZE,
        background_color="#f5f5f7",
    )
    webview.start()
    return True


def run_edge(url: str) -> bool:
    edge = find_edge()
    if edge is None:
        return False

    # 独立 profile，避免复用已开的 Edge 进程导致窗口关不掉或提前返回
    profile = ROOT / "data" / "edge_profile"
    profile.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(edge),
        f"--app={url}",
        f"--user-data-dir={profile}",
        f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}",
        "--disable-background-mode",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    subprocess.run(cmd, check=False)
    return True


def run_browser(url: str) -> None:
    webbrowser.open(url)
    print("已在默认浏览器打开。关闭此窗口后服务才会停止。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


# ── 入口 ──────────────────────────────────────────────────────────────────────


def check(host: str, port: int) -> int:
    print("== 桌面模式检查 ==")
    print(f"pywebview : {'已安装' if has_webview() else '未安装（pip install pywebview 可启用原生窗口）'}")
    edge = find_edge()
    print(f"Edge      : {edge if edge else '未找到'}")
    print(f"端口 {port} : {'已占用（会直接复用现有服务）' if port_open(host, port) else '空闲'}")
    print()
    if has_webview():
        print("推荐: python desktop.py --mode webview")
    elif edge:
        print("推荐: python desktop.py --mode edge")
    else:
        print("推荐: python desktop.py --mode browser")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="以桌面窗口方式启动 imgsearch")
    parser.add_argument("--mode", choices=["auto", "webview", "edge", "browser"], default="auto")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--check", action="store_true", help="只检查环境")
    args = parser.parse_args()

    from imgsearch import Config

    cfg = Config.from_yaml()
    host = args.host or cfg.app.host
    port = int(args.port or cfg.app.port)
    window_title = cfg.app.title or WINDOW_TITLE
    url = f"http://{host}:{port}"

    if args.check:
        return check(host, port)

    redirect_output_when_no_console()

    server = None
    if port_open(host, port):
        print(f"检测到 {url} 已在运行，直接打开窗口。")
    else:
        print(f"正在启动服务 {url} ...")
        server = start_server(host, port)
        if not wait_port(host, port):
            print("服务启动超时，请检查 logs/desktop.log")
            return 1

    try:
        if args.mode in ("auto", "webview"):
            if run_webview(url, window_title):
                return 0
            if args.mode == "webview":
                print("未安装 pywebview。先执行: pip install pywebview")
                return 1

        if args.mode in ("auto", "edge"):
            if run_edge(url):
                return 0
            if args.mode == "edge":
                print("未找到 Microsoft Edge。")
                return 1

        run_browser(url)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if server is not None:
            try:
                server.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
