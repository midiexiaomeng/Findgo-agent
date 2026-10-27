# 开发笔记

## 当前状态

- 测试：63 个，覆盖向量库契约、LLM 解析与降级、Planner、Memory、Critic、Responder、Orchestrator
- 端到端：已用 DeepSeek `deepseek-flash` + 1867 张 COCO 图片验证通过
- 桌面端：PyQt5，已通过启动测试

## 已知问题

**ChromaDB 1.5.9 在 Windows 上 HNSW 索引写入异常。**
现象：索引完成后 `data/chromadb` 下只有 `chroma.sqlite3` 和 `index_metadata.pickle`，缺少 `data_level0.bin`，再次查询报 `Error loading hnsw index`。
结论：当前环境改用 Qdrant 本地模式，稳定。

**8GB 显存同时承载 CLIP 与 3B 模型会触发显存交换。**
现象：重排耗时从 10 秒级涨到 4 分钟级。
处理：把 `embedding.device` 设为 `cpu`，CLIP 只编码查询文本，几十毫秒；LLM 独占显存。

**Gradio 6 的 `css` / `theme` 从 `Blocks()` 移到了 `launch()`。**
写错位置不报错，样式静默失效。

**transformers 4.57 的 Chinese-CLIP `get_text_features` 有回归 bug。**
`add_pooling_layer=False` 但 `get_text_features` 仍读 `pooler_output`，得到 `None` 后抛 `TypeError`。
处理：`clip_encoder.py` 里检测到 pooler 为空时，改取最后一层 `[CLS]` 再过 `text_projection`。

## 调试技巧

**按 trace_id 查一轮完整动作：**

```python
import sqlite3, json
conn = sqlite3.connect("data/agent_sessions.sqlite")
rows = conn.execute(
    "SELECT trace_id, action, tool_name, tool_args, tool_result FROM traces WHERE session_id = ?",
    ("你的session_id",),
).fetchall()
```

**验证 LLM 连通：**

```powershell
python scripts\test_agent_llm.py
```

**验证完整链路：**

```powershell
python scripts\test_agent_e2e.py "帮我找几张猫的图片"
```

**检查检索质量（需要 COCO caption 标注）：**

```powershell
python scripts\evaluate.py --captions data\captions_val2017.json --n 100
```

## 待办

- [ ] 以图搜图工具（`search_by_image` 注册到工具层）
- [ ] 混合检索：BM25 + 向量，RRF 融合
- [ ] 中文图像描述：用 Qwen2.5-VL 替代 BLIP
- [ ] Recall@K 实测与对比实验
- [ ] PyInstaller 打包 exe
- [ ] 多轮对话：把 Memory 的历史更完整地喂给 Planner

## 开发约定

- 配置改动只提交 `*.example.yaml`，本地 `agent.yaml` 不入库
- 新增工具只需继承 `BaseTool` 并在 `configs/tools.yaml` 注册
- 每次提交前跑 `pytest tests -q`
- 代码风格：black + ruff，行宽 100
