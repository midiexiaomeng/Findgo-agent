# imgsearch-agent

基于 **ReAct** 的多模态图像检索 Agent，带三层记忆与动态工具扩展。

它不是「输入查询 → 返回结果」的被动检索，而是会自己判断：该用哪个工具、结果够不够可信、要不要改写查询重试、本地不够时要不要联网补图，最后用自然语言把结果讲清楚。

图像与文本用 Chinese-CLIP 映射到同一语义空间，向量库负责快速召回，BLIP 生成图像描述供大模型做语义重排。所有图像处理都在本机完成，只有可选的 LLM 重排会调用 API。

## 核心能力

- **以文搜图**：中文语义对齐，支持限定目录、排除已返回结果、相似图去重
- **以图搜图**：给一张本地图片找视觉相似的图，支持直接粘贴
- **三级检索**：跨模态嵌入 → 向量召回 → 大模型重排，大批量时只重排头部
- **双向量库**：ChromaDB 与 Qdrant 共用一套接口，切换前自检失败会回滚
- **联网补图**：本地不够时按需下载，缓存在独立目录，有独立容量上限
- **三层记忆**：短期会话 / 摘要压缩 / 跨会话向量记忆，外加结构化事实
- **动态工具**：会话级临时工具，源码过安全扫描、人工审核后才可执行
- **安全沙箱**：素材只读、workspace 读写、系统目录拒绝，删除必经暂存区
- **桌面应用**：PyQt5 无边框窗口，不占端口、不依赖浏览器

## 架构

```
┌─────────────────────────────────────────────────────┐
│ 交互层   PyQt5 桌面应用（ui/）/ CLI / 可选 Gradio      │
├─────────────────────────────────────────────────────┤
│ Agent 层 Orchestrator                                │
│          ├─ Planner        任务拆解 + 路径识别        │
│          ├─ ToolExecutor   校验 / 超时 / 截断 / 权限  │
│          ├─ Critic         置信校验 + 改写重试         │
│          ├─ Responder      结果合成                   │
│          └─ Memory         会话 / 事实 / 摘要 / 轨迹   │
├─────────────────────────────────────────────────────┤
│ 工具层   16 个 BaseTool，统一注册与执行                │
├─────────────────────────────────────────────────────┤
│ 内核层   Pipeline / Searcher / Indexer / Encoder      │
│          VectorStore（Qdrant / Chroma）               │
├─────────────────────────────────────────────────────┤
│ 基础层   Sandbox / Config / Logger / Theme            │
└─────────────────────────────────────────────────────┘
```

### 工具清单

| 工具 | 作用 | 权限 |
| --- | --- | --- |
| `image_retrieval` | 文本检索，支持 `folder_path`、`all_results`、`exclude_paths`、相似去重 | read |
| `image_search_by_image` | 以图搜图 | read |
| `find_duplicates` | 按向量相似度找重复图 | read |
| `filter_results` | 对已有结果二次过滤（暂存、分数、关键词） | read |
| `describe_image` | 单图 BLIP 描述 | read |
| `folder_scan` | 扫描目录统计图片数 | read |
| `list_indexed_folders` | 列出已索引目录 | read |
| `count_indexed_images` | 索引内数量统计（逻辑视图） | read |
| `get_index_stats` | 数量、后端、设备 | read |
| `preview_staged` | 只读查看暂存区清单 | read |
| `index_folder` | 建索引，mtime 缓存 + 文件夹锁 | index_write |
| `remove_index` | 删索引，默认 dry-run，支持清孤儿向量 | index_write |
| `switch_vector_backend` | 切后端，自检失败回滚 | index_write |
| `export_results` | 导出 CSV / TXT | file_write |
| `trash_manage` | 暂存 / 恢复 / 永久删除 / 清理 | file_write |
| `web_image_search` | 联网搜索下载并建索引 | file_write |

`trash_manage` 的 `purge` 动作单独提升为 `danger_delete`，Agent 无法自主调用，必须界面人工确认。

## 三层记忆

```
短期会话   messages 表，按 session_id 隔离，保留最近 N 轮
摘要压缩   超阈值时 LLM 压缩成事实摘要，原文保留
结构化事实 每会话一份 JSON，是本会话唯一权威约束源
长期记忆   Qdrant 存跨会话稳定偏好，只作参考不覆盖事实
```

优先级：**结构化事实 > 最近对话 > 摘要 > 向量记忆**。

事实通过 patch 增量更新，`null` 表示删除字段：

```python
memory.patch_facts({"count": 3})
memory.patch_facts({"filters": {"color": None}})   # 删除颜色限制
memory.clear_facts()
```

交互式对话每轮自动更新事实；批处理 `run_tasks` 默认只读不改写，避免多条独立任务互相污染。

## 安全边界

所有路径操作走同一个 `Sandbox`，分三类：

| 区域 | 读 | 写 / 删 |
| --- | --- | --- |
| `image_roots`（datasets、Pictures） | 允许 | 禁止 |
| `workspace_roots`（data 目录） | 允许 | 允许 |
| 系统目录（Windows、Program Files） | 禁止 | 禁止 |

符号链接一律拒绝，防止绕过白名单。

删除流程：

```
stage   → 硬链接进暂存区 + 向量打 is_staged 标记，原图不动
restore → 删链接 + 标记复原
purge   → 删原图 + 删链接 + 向量移除（需人工确认）
cleanup → 按留存天数与容量上限自动清理
```

## 目录结构

```
src/imgsearch_agent/
├── agent/            Agent 层：编排、规划、执行、记忆、动态工具
├── retriever/        内核层：CLIP、索引、检索、向量库、重排
├── tools/            16 个工具实现
├── ui/               界面组件（只渲染，不写业务）
│   ├── agent_runner.py     发送、后台线程、任务链重跑
│   ├── session_panel.py    会话管理
│   ├── file_tree.py        目录浏览与预览
│   ├── bubble.py           消息气泡与图库
│   ├── title_bar.py        自定义标题栏
│   ├── input_panel.py      输入框与拖拽手柄
│   ├── skills_panel.py     Skills 面板
│   ├── dynamic_dialog.py   动态工具审核
│   ├── chat_render.py      路径高亮与消息 HTML
│   └── thumbnail_cache.py  缩略图缓存
├── utils/            沙箱、schema、日志
├── memory_engine 相关已合并到 agent/memory.py
├── long_term.py      Qdrant 长期记忆
├── trash.py          暂存区
├── downloads.py      下载缓存
├── diagnostics.py    环境自检
└── settings_dialog.py 设置面板

desktop_app.py        启动入口，只做窗口组装
```

## 快速开始

```powershell
# 1. 安装依赖
pip install -r requirements-core.txt

# 2. 下载 CLIP + BLIP（约 1.6GB，走 hf-mirror）
python scripts\download_models.py

# 3. 生成 Agent 配置
python scripts\init_agent_config.py

# 4. 配置 API Key（环境变量优先，不写进文件）
setx DEEPSEEK_API_KEY "sk-你的key"
# 重开终端后生效

# 5. 建立索引
python cli.py index "D:\你的图片目录"

# 6. 启动桌面应用
start-app.bat
```

网页版：`python run.py`（7861 端口）。

## 示例对话

```
你：在 D:\val2017 里找 40 张含猫的图片

Agent：
[任务链]
📁 打开 D:\val2017
🔍 40 张含猫的图片

已为你找到 40 张猫的图片。
[前 20 张缩略图 + 右下角省略号入口]
```

如果第一次检索置信度低于阈值，Agent 会自动改写查询再试；点数超过 20 张时气泡右下角出现省略号，点开是完整图库，悬停预览、双击放大。

## 配置

| 文件 | 作用 | 是否入库 |
|---|---|---|
| `configs/agent.yaml` | Agent 行为、提示词、LLM、沙箱、暂存区 | 否（本地配置） |
| `configs/agent.example.yaml` | 同上模板 | 是 |
| `configs/tools.yaml` | 工具注册 | 是 |
| `configs/custom_tools.yaml` | 用户保存的动态工具 | 否 |
| `config.yaml` | 检索模型、向量库、重排、下载缓存 | 是 |

关键项：

```yaml
agent:
  max_retries: 2              # Critic 低置信重试次数
  similarity_threshold: 0.25  # 低于此值判定低置信

sandbox:
  image_roots: [./datasets]   # 只读素材
  workspace_roots: [./data]   # 可读写

trash_staging:
  path: ./data/trash_staging
  max_keep_days: 14
  max_total_size_gb: 10

download_cache:
  path: ./data/downloads
  max_total_size_gb: 5
```

## 测试

```powershell
pytest tests -q
```

当前 225 个测试，覆盖沙箱、暂存区、记忆、工具参数校验、检索链路、动态工具安全闸门。

## Roadmap

- [x] 三级检索：CLIP 嵌入 + 向量召回 + LLM 重排
- [x] 向量库抽象层，ChromaDB / Qdrant 双后端
- [x] ReAct 编排：Planner / Memory / Critic / Responder
- [x] PyQt5 桌面应用，无边框窗口与自定义标题栏
- [x] 16 个 Agent 工具，权限分组与统一执行器
- [x] 路径识别：正则优先 + LLM 辅助
- [x] 分域沙箱与暂存区软删除
- [x] 三层记忆 + 结构化事实
- [x] 动态工具加载与安全闸门
- [x] 联网补图与下载缓存
- [x] 225 个单元测试 + CI 配置
- [ ] 混合检索：BM25 + 向量，RRF 融合
- [ ] 中文图像描述（Qwen2.5-VL 替代 BLIP）
- [ ] Recall@K 实测与对比实验
- [ ] PyInstaller 打包 exe

## 贡献指南

新增一个内置工具：

1. 在 `src/imgsearch_agent/tools/` 下新建类，继承 `BaseTool`
2. 定义 `name`、`description`、`args_schema`、`permission`、`timeout`、`run()`
3. `tools/__init__.py` 导出
4. `orchestrator.py` 的 `self.tools` 注册
5. `configs/tools.yaml` 登记
6. 补测试

临时需求不必新增工具，可以走动态工具：Planner 生成 `DynamicToolSpec`，源码过黑名单扫描、在隔离命名空间编译，人工审核后执行。

## 已知限制

- **描述是英文的**。BLIP base 只支持英文，中文查询与英文描述之间有语义落差。
- **只做稠密向量召回**。对「图中文字」「精确专有名词」支持有限。
- **ChromaDB 在部分 Windows 环境写入异常**，实测 Qdrant 更稳。
- **8GB 显存**同时跑 CLIP 与 3B 模型会触发显存交换，建议 LLM 走 API。
- **联网搜索依赖 Bing 页面结构**，字段调整会导致下载失败，后续可换官方 API。
- **尚未完成 Recall@K 定量评估**。

## License

MIT
