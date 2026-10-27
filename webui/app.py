"""Gradio 界面。

设计目标：打开就能用，不需要读文档。
- 搜索框 + 一个按钮，结果直接以图片墙呈现
- 图片下方给完整路径、三路分数（向量 / LLM / 融合）和图像描述
- 索引、切库、体检都放在二级页面，不干扰主流程
"""

from __future__ import annotations

import html
import logging
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from imgsearch_agent.retriever import __version__
from imgsearch_agent.retriever.config import Config
from imgsearch_agent.retriever.pipeline import Pipeline
from imgsearch_agent.retriever.runtime import describe_device, human_time

logger = logging.getLogger(__name__)

_PIPELINE: Pipeline | None = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            logger.info("初始化 Pipeline ...")
            _PIPELINE = Pipeline(Config.from_yaml())
    return _PIPELINE


# ── 样式 ──────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }

#app-header {
    padding: 18px 22px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(99,102,241,.10), rgba(14,165,233,.08));
    border: 1px solid var(--border-color-primary, #e5e7eb);
    margin-bottom: 14px;
}
#app-header .title { font-size: 24px; font-weight: 700; letter-spacing: .5px; }
#app-header .sub { font-size: 13px; opacity: .72; margin-top: 4px; }
#app-header .badges { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; background: rgba(127,127,127,.12);
    border: 1px solid rgba(127,127,127,.18);
}
.badge.ok { background: rgba(22,163,74,.12); border-color: rgba(22,163,74,.3); }
.badge.warn { background: rgba(217,119,6,.12); border-color: rgba(217,119,6,.3); }

.status-line { font-size: 13px; opacity: .8; margin: 4px 0 8px 2px; }

.results { display: flex; flex-direction: column; gap: 8px; margin-top: 6px; }
.result-item {
    display: grid; grid-template-columns: 40px minmax(0,1fr) auto;
    gap: 10px; align-items: start;
    padding: 10px 14px; border-radius: 10px;
    border: 1px solid var(--border-color-primary, #e5e7eb);
    background: var(--background-fill-primary, #fff);
}
.result-item .rank {
    font-weight: 700; font-size: 15px; opacity: .55;
    padding-top: 1px;
}
.result-item .name { font-weight: 600; font-size: 14px; }
.result-item .path {
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 11.5px; opacity: .62; word-break: break-all; margin-top: 2px;
}
.result-item .desc { font-size: 12.5px; line-height: 1.5; margin-top: 5px; opacity: .85; }
.result-item .score { font-size: 11.5px; text-align: right; white-space: nowrap; opacity: .75; line-height: 1.7; }
.result-item .score b { font-size: 13px; opacity: 1; }
.missing { color: #dc2626; font-weight: 600; }
.empty { padding: 34px; text-align: center; opacity: .5; font-size: 14px; }

.kv { width: 100%; border-collapse: collapse; font-size: 13px; }
.kv td { padding: 7px 10px; border-bottom: 1px solid var(--border-color-primary, #eee); vertical-align: top; }
.kv td:first-child { width: 170px; opacity: .65; }
.kv code { font-size: 12px; word-break: break-all; }
.hint { font-size: 12.5px; opacity: .65; line-height: 1.6; }
"""


# ── 渲染 ──────────────────────────────────────────────────────────────────────


def _badges() -> str:
    try:
        p = get_pipeline()
        stats = p.stats()
        health = p.health()
        backend = stats["backend"]
        count = stats["images"]
        device = describe_device(p.encoder.device) if p.encoder.loaded else health["device"]
        llm_cls = "ok" if health["llm_ok"] else "warn"
        llm_text = "LLM 就绪" if health["llm_ok"] else "LLM 未连接"
        return (
            f'<span class="badge">向量库 {backend} · {count} 张</span>'
            f'<span class="badge">设备 {html.escape(device)}</span>'
            f'<span class="badge {llm_cls}">{llm_text}</span>'
            f'<span class="badge">v{__version__}</span>'
        )
    except Exception as e:
        return f'<span class="badge warn">初始化失败: {html.escape(str(e))}</span>'


def _header() -> str:
    title = "多模态图像检索"
    try:
        title = get_pipeline().cfg.app.title or title
    except Exception:
        pass
    return f"""
    <div id="app-header">
      <div class="title">{html.escape(title)}</div>
      <div class="sub">Chinese-CLIP 跨模态嵌入 · 向量召回 · Qwen2.5 语义精排 · 全本地运行</div>
      <div class="badges">{_badges()}</div>
    </div>
    """


def _gallery_value(results: list) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for r in results:
        if not r.exists:
            continue
        label = f"#{r.rank}  {r.filename}  ({r.score:.2f})"
        items.append((r.path, label))
    return items


def _details_html(results: list) -> str:
    if not results:
        return '<div class="empty">没有结果</div>'

    rows = []
    for r in results:
        path_html = html.escape(r.path)
        if not r.exists:
            path_html = (
                f'<span class="missing">文件已丢失</span> '
                f'<span style="opacity:.5">{path_html}</span>'
            )
        llm = "—" if r.llm_score is None else f"{r.llm_score:.2f}"
        desc = html.escape(r.caption) if r.caption else '<span style="opacity:.45">无描述</span>'
        rows.append(
            f"""
            <div class="result-item">
              <div class="rank">#{r.rank}</div>
              <div>
                <div class="name">{html.escape(r.filename or "(未命名)")}</div>
                <div class="path">{path_html}</div>
                <div class="desc">{desc}</div>
              </div>
              <div class="score">
                <b>{r.score:.3f}</b><br/>
                向量 {r.vec_score:.3f} · LLM {llm}
              </div>
            </div>
            """
        )
    return f'<div class="results">{"".join(rows)}</div>'


def _status_md(response, extra: str = "") -> str:
    if response is None:
        return extra
    line = response.status_line()
    if response.mode == "image" and response.query_caption:
        line += f"  \n查询图描述：*{html.escape(response.query_caption)}*"
    if response.note:
        line += f"  \n<span style='opacity:.6;font-size:12px'>{html.escape(response.note)}</span>"
    if extra:
        line += f"  \n{extra}"
    return line


def _error_md(message: str) -> str:
    return f"**出错了**：{html.escape(message)}"


# ── 回调 ──────────────────────────────────────────────────────────────────────


def do_text_search(query: str, top_k: int, recall_k: int, use_llm: bool):
    query = (query or "").strip()
    if not query:
        return "请输入检索内容", [], '<div class="empty">等待输入</div>', _header()

    try:
        p = get_pipeline()
        loading = "" if p.encoder.loaded else "首次检索正在加载 CLIP 模型，请稍候…"
        response = p.search_text(
            query,
            top_k=int(top_k),
            recall_k=int(recall_k),
            use_rerank=bool(use_llm),
        )
        return (
            _status_md(response, loading),
            _gallery_value(response.results),
            _details_html(response.results),
            _header(),
        )
    except Exception as e:
        logger.exception("以文搜图失败")
        return _error_md(str(e)), [], "", _header()


def do_image_search(image_path: str | None, top_k: int, recall_k: int, use_llm: bool):
    if not image_path:
        return "请先上传一张参考图片", [], '<div class="empty">等待上传</div>', _header()

    try:
        p = get_pipeline()
        response = p.search_image(
            image_path,
            top_k=int(top_k),
            recall_k=int(recall_k),
            use_rerank=bool(use_llm),
        )
        return (
            _status_md(response),
            _gallery_value(response.results),
            _details_html(response.results),
            _header(),
        )
    except Exception as e:
        logger.exception("以图搜图失败")
        return _error_md(str(e)), [], "", _header()


def do_index(
    folder: str,
    recursive: bool,
    reset: bool,
    with_caption: bool,
    progress: gr.Progress = gr.Progress(),
):
    folder = (folder or "").strip().strip('"')
    if not folder:
        return "请填写图片文件夹路径", "", _header()

    path = Path(folder).expanduser()
    if not path.exists() or not path.is_dir():
        return f"文件夹不存在：{folder}", "", _header()

    if reset:
        progress(0.0, desc="清空当前向量库…")

    def cb(done: int, total: int, name: str) -> None:
        progress(min(done / max(total, 1), 1.0), desc=f"{done}/{total}  {name}")

    try:
        p = get_pipeline()
        stats = p.index_folder(
            path,
            recursive=bool(recursive),
            reset=bool(reset),
            with_caption=bool(with_caption),
            progress=cb,
        )
        total_in_db = p.store.count()
        msg = f"**索引完成** · {stats.summary()}"
        if stats.errors:
            msg += f"  \n<span style='opacity:.6;font-size:12px'>前几条错误：{html.escape('; '.join(stats.errors[:3]))}</span>"
        rows = [
            ("本次扫描", f"{stats.total} 张"),
            ("新增入库", f"{stats.indexed} 张"),
            ("跳过（已存在）", f"{stats.skipped} 张"),
            ("失败", f"{stats.failed} 张"),
            ("生成描述", f"{stats.captioned} 条"),
            ("清理失效记录", f"{stats.pruned} 条"),
            ("耗时", human_time(stats.elapsed)),
            ("库内总数", f"{total_in_db} 张"),
        ]
        table = "<table class='kv'>" + "".join(
            f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>" for k, v in rows
        ) + "</table>"
        return msg, table, _header()
    except Exception as e:
        logger.exception("索引失败")
        return _error_md(str(e)), "", _header()


def do_switch_backend(backend: str):
    if not backend:
        return "请选择后端", _settings_html()
    try:
        p = get_pipeline()
        msg = p.switch_backend(backend)
        return msg, _settings_html()
    except Exception as e:
        logger.exception("切换向量库失败")
        return f"切换失败：{html.escape(str(e))}", _settings_html()


def _settings_html() -> str:
    try:
        p = get_pipeline()
    except Exception as e:
        return f"<p class='hint'>初始化失败：{html.escape(str(e))}</p>"

    cfg = p.cfg
    stats = p.stats()
    health = p.health()

    vs = cfg.vector_store
    if stats["backend"] == "qdrant":
        where = (
            f"local · {cfg.resolve(vs.qdrant.path)}"
            if vs.qdrant.mode == "local"
            else f"server · {vs.qdrant.url}"
        )
    else:
        where = str(cfg.resolve(vs.chromadb.path))

    rows = [
        ("配置文件", str(cfg.root / "config.yaml")),
        ("运行设备", html.escape(health["device"])),
        ("CLIP 模型", html.escape(cfg.embedding.model)),
        ("嵌入维度", str(stats["dim"])),
        ("向量库", f"{stats['backend']} · {where}"),
        ("collection", stats["collection"]),
        ("库内图片", f"{stats['images']} 张"),
        ("LLM 重排", f"{html.escape(p.reranker.client.describe_backend())} · {'在线' if health['llm_ok'] else '未连接'}"),
        ("LLM 状态", html.escape(health["llm_message"])),
        ("BLIP 描述", f"{'启用' if cfg.caption.enabled else '关闭'} · 缓存 {p.captioner.cache_size()} 条"),
        ("召回 / 精排 / 展示", f"{cfg.search.recall_k} / {cfg.rerank.max_candidates} / {cfg.search.top_k}"),
        ("融合权重", f"向量 {cfg.search.fusion.vector_weight} · LLM {cfg.search.fusion.llm_weight}"),
    ]
    table = "<table class='kv'>" + "".join(
        f"<tr><td>{k}</td><td><code>{v}</code></td></tr>" for k, v in rows
    ) + "</table>"
    return table


# ── 构建 ──────────────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    try:
        app_title = get_pipeline().cfg.app.title or "多模态图像检索"
    except Exception:
        app_title = "多模态图像检索"

    blocks_kwargs: dict[str, Any] = {"title": app_title}
    # Gradio 4/5 的主题和 CSS 挂在 Blocks 上，6 挪到了 launch
    if _major_version() < 6:
        blocks_kwargs.update(theme=gr.themes.Soft(), css=CSS)

    with gr.Blocks(**blocks_kwargs) as demo:
        header = gr.HTML(_header())

        # ── 以文搜图 ──
        with gr.Tab("以文搜图"):
            with gr.Row():
                query = gr.Textbox(
                    show_label=False,
                    placeholder="描述你想找的画面，例如：夕阳下的海边、一只橘猫趴在窗台上、城市夜景",
                    scale=8,
                    container=False,
                )
                text_btn = gr.Button("搜索", variant="primary", scale=1, min_width=100)

            with gr.Accordion("检索参数", open=False):
                with gr.Row():
                    t_topk = gr.Slider(4, 40, value=12, step=1, label="返回数量")
                    t_recall = gr.Slider(10, 200, value=50, step=10, label="向量召回数")
                    t_llm = gr.Checkbox(value=True, label="LLM 语义重排")

            text_status = gr.Markdown("")
            text_gallery = gr.Gallery(
                label="检索结果",
                columns=4,
                height="auto",
                object_fit="cover",
                allow_preview=True,
            )
            text_details = gr.HTML('<div class="empty">输入描述开始检索</div>')

            outputs = [text_status, text_gallery, text_details, header]
            text_btn.click(
                do_text_search, [query, t_topk, t_recall, t_llm], outputs
            )
            query.submit(do_text_search, [query, t_topk, t_recall, t_llm], outputs)

        # ── 以图搜图 ──
        with gr.Tab("以图搜图"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    query_image = gr.Image(
                        type="filepath",
                        label="参考图片",
                        height=300,
                        sources=["upload", "clipboard"],
                    )
                    image_btn = gr.Button("开始搜索", variant="primary")
                    gr.Markdown(
                        "<div class='hint'>查询图会先用 BLIP 生成描述，"
                        "再用这段描述驱动 LLM 对候选做语义精排。"
                        "所以「以图搜图」的结果不只看长得像，也看语义是否一致。</div>"
                    )
                with gr.Column(scale=3):
                    with gr.Accordion("检索参数", open=False):
                        with gr.Row():
                            i_topk = gr.Slider(4, 40, value=12, step=1, label="返回数量")
                            i_recall = gr.Slider(10, 200, value=50, step=10, label="向量召回数")
                            i_llm = gr.Checkbox(value=True, label="LLM 语义重排")
                    image_status = gr.Markdown("")
                    image_gallery = gr.Gallery(
                        label="检索结果",
                        columns=4,
                        height="auto",
                        object_fit="cover",
                        allow_preview=True,
                    )

            image_details = gr.HTML('<div class="empty">上传参考图开始搜索</div>')
            image_outputs = [image_status, image_gallery, image_details, header]
            image_btn.click(
                do_image_search, [query_image, i_topk, i_recall, i_llm], image_outputs
            )

        # ── 索引 ──
        with gr.Tab("建立索引"):
            with gr.Row():
                folder = gr.Textbox(
                    label="图片文件夹",
                    placeholder=r"例如 D:\Photos 或 ./my_images",
                    scale=4,
                )
                index_btn = gr.Button("开始索引", variant="primary", scale=1, min_width=110)

            with gr.Row():
                recursive = gr.Checkbox(value=True, label="包含子文件夹")
                reset = gr.Checkbox(value=False, label="先清空当前库再索引")
                with_caption = gr.Checkbox(value=True, label="生成图像描述（LLM 重排需要）")

            gr.Markdown(
                "<div class='hint'>描述生成只在索引时做一次，之后存进向量库和本地缓存；"
                "重建索引时会直接复用缓存，不会重复跑模型。"
                "第一次索引会加载 CLIP 和 BLIP，耗时较长，之后都是增量索引。</div>"
            )

            index_status = gr.Markdown("")
            index_table = gr.HTML("")
            index_btn.click(
                do_index,
                [folder, recursive, reset, with_caption],
                [index_status, index_table, header],
            )

        # ── 设置 ──
        with gr.Tab("设置与诊断"):
            gr.Markdown("### 当前配置")
            settings_box = gr.HTML(_settings_html())

            gr.Markdown("### 切换向量库")
            with gr.Row():
                backend_dd = gr.Dropdown(
                    ["chromadb", "qdrant"],
                    value="chromadb",
                    label="向量库后端",
                    scale=2,
                )
                switch_btn = gr.Button("切换", scale=1, min_width=90)
                refresh_btn = gr.Button("刷新状态", scale=1, min_width=90)

            switch_msg = gr.Markdown(
                "<div class='hint'>ChromaDB 和 Qdrant 各自管理自己的数据，"
                "切换后需要重新建立索引。代码层面两者走同一套 VectorStore 接口。</div>"
            )

            switch_btn.click(do_switch_backend, [backend_dd], [switch_msg, settings_box])
            refresh_btn.click(
                lambda: ("状态已刷新", _settings_html()),
                [],
                [switch_msg, settings_box],
            )
            refresh_btn.click(lambda: _header(), [], [header])

        demo.load(lambda: (_header(), _settings_html()), [], [header, settings_box])

    demo.queue(default_concurrency_limit=1)
    return demo


def _major_version() -> int:
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:
        return 4


def launch(**overrides: Any):
    cfg = Config.from_yaml()
    demo = build_app()
    kwargs: dict[str, Any] = {
        "server_name": cfg.app.host,
        "server_port": int(cfg.app.port),
        "inbrowser": bool(cfg.app.open_browser),
        "share": bool(cfg.app.share),
        "show_error": True,
    }
    # Gradio 6 把 theme/css 挪到了 launch；4/5 仍在 Blocks 上
    if _major_version() >= 6:
        kwargs["theme"] = gr.themes.Soft()
        kwargs["css"] = CSS
    kwargs.update(overrides)
    return demo.launch(**kwargs)


def create_app() -> gr.Blocks:
    return build_app()
