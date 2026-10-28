"""对话式 Agent 界面（Gradio）。

左侧对话，右侧图片墙。每次用户发言由 Agent 自主决策、检索、总结。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import gradio as gr

from .agent import FindgoAgent
from .config import AgentConfig

logger = logging.getLogger("findgo.app")

_AGENT: FindgoAgent | None = None
_LOCK = threading.Lock()

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
#header {
    padding: 16px 20px; border-radius: 14px; margin-bottom: 12px;
    background: linear-gradient(135deg, rgba(99,102,241,.10), rgba(14,165,233,.08));
    border: 1px solid var(--border-color-primary, #e5e7eb);
}
#header .title { font-size: 22px; font-weight: 700; }
#header .sub { font-size: 13px; opacity: .7; margin-top: 4px; }
.status-line { font-size: 12.5px; opacity: .7; margin-top: 6px; }
.hint { font-size: 12.5px; opacity: .6; line-height: 1.6; }
"""


def get_agent() -> FindgoAgent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    with _LOCK:
        if _AGENT is None:
            config = AgentConfig()
            logger.info("初始化 Agent: %s", config.config_path)
            _AGENT = FindgoAgent(config=config)
    return _AGENT


def _gallery_value(results: list) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for img in results:
        if isinstance(img, dict):
            path = img.get("path", "")
            score = img.get("score", 0.0)
        else:
            path = getattr(img, "path", "")
            score = getattr(img, "score", 0.0)
        if path and Path(path).exists():
            items.append((path, f"{float(score):.2f}"))
    return items


def respond(message: str, history: list[dict]):
    message = (message or "").strip()
    if not message:
        return "", history, [], ""

    try:
        agent = get_agent()
        result = agent.run(message)
    except Exception as e:
        logger.exception("Agent 执行失败")
        history = list(history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"出错了：{e}"},
        ]
        return "", history, [], ""

    history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.reply},
    ]
    status = (
        f"工具 `{result.tool or '无'}` · 置信度 {result.confidence:.3f}"
        f" · 重试 {result.retries} · trace `{result.trace_id}`"
    )
    return "", history, _gallery_value(result.results), status


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Findgo") as demo:
        gr.HTML(
            """
            <div id="header">
              <div class="title">Findgo</div>
              <div class="sub">基于 ReAct 的多模态图像检索智能体 · 自主决策 · 检索 · 校验 · 总结</div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2, min_width=380):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=520,
                    layout="bubble",
                    placeholder="问我：帮我找几张猫的图片",
                )
                with gr.Row():
                    message = gr.Textbox(
                        show_label=False,
                        placeholder="描述你想找的图片，回车发送",
                        scale=5,
                    )
                    send = gr.Button("发送", variant="primary", scale=1, min_width=80)
                status = gr.Markdown("")

            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    label="检索结果",
                    columns=3,
                    height="auto",
                    object_fit="cover",
                    allow_preview=True,
                )
                gr.Markdown(
                    "<div class='hint'>首次对话会加载 CLIP 与向量库，约 10 到 20 秒；"
                    "之后每次对话只需几秒。Agent 会自主判断是否需要检索、结果是否可信。</div>"
                )

        send.click(respond, [message, chatbot], [message, chatbot, gallery, status])
        message.submit(respond, [message, chatbot], [message, chatbot, gallery, status])

    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    config = AgentConfig()
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        show_error=True,
        css=CSS,
    )


if __name__ == "__main__":
    main()
