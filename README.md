# imgsearch-agent

基于 **ReAct** 的多模态图像检索 Agent，带三层记忆与动态工具扩展。

它不是「输入查询 → 返回结果」的被动检索，而是会自己判断：该用哪个工具、结果够不够可信、要不要改写查询重试、本地不够时要不要联网补图，最后用自然语言把结果讲清楚。

图像与文本用 Chinese-CLIP 映射到同一语义空间，向量库负责快速召回，BLIP 生成图像描述供大模型做语义重排。所有图像处理都在本机完成，只有可选的 LLM 重排会调用 API。

## 前置环境要求

先确认自己能跑起来，再往下看。

| 项目 | 最低 | 推荐 |
| --- | --- | --- |
| 系统 | Windows 10 / Linux | Windows 11 |
| Python | 3.9 | 3.11 |
| 内存 | 8 GB | 16 GB |
| 显存 | 可选（可用 CPU） | 6 GB 以上，CLIP 走 GPU |
| 磁盘 | 10 GB | 30 GB（模型 1.6GB + 向量库 + 图片缓存） |

**关于「能不能完全离线」：**

- **CLIP / BLIP 本地跑**：首次需联网下载模型（约 1.6GB），之后完全离线
- **LLM 重排必须联网**：默认走 DeepSeek API，需要 `DEEPSEEK_API_KEY`
- **想完全离线**：把 `config.yaml` 的 `rerank.backend` 改成 `ollama`，本地起 Ollama 服务；或改成 `transformers` 直接加载本地权重（需要额外几 GB 显存）

没有 API Key 也能用，只是会跳过重排，退回纯向量排序，效果会差一些。

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

### 权限等级说明

| 等级 | 能做什么 | 谁会触发 |
| --- | --- | --- |
| `read` | 只读文件、查向量库、统计 | 检索、预览、统计类工具 |
| `index_write` | 修改向量库内容 | 建索引、删索引、切后端 |
| `file_write` | 在 `data/` 下写文件 | 导出、暂存、联网下载 |
| `danger_delete` | 删原图 | 仅 `purge`，必须人工确认 |

前三级 Agent 可以自主调用，`danger_delete` 会被 `ToolExecutor` 直接拦下，返回 `need_user_confirm`。

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
stage   → 硬链接进暂存区 + 向量打 is_staged 标记，原图不动   【Agent 自动】
restore → 删链接 + 标记复原                                  【Agent 或用户】
purge   → 删原图 + 删链接 + 向量移除                          【必须界面人工确认】
cleanup → 按留存天数与容量上限自动清理                        【后台定时 / 手动】
```

Agent 能做的只有 `stage` 和 `restore`。`purge` 是唯一会真正删除原图的操作，被三层拦截：工具层拒绝、执行器拒绝、界面二次确认。

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

### 配置优先级

```
环境变量  >  configs/agent.yaml  >  代码默认值
```

环境变量格式是 `IMGSRCH_<SECTION>__<KEY>`，比如 `IMGSRCH_VECTOR_STORE__QDRANT__PATH` 会覆盖 Qdrant 路径。适合 Docker 或临时调试，不用改文件。

`configs/agent.yaml` 不进 git（含 LLM 地址与个人路径），首次运行由 `init_agent_config.py` 从模板生成。

## 故障排查

### 日志在哪

```
logs/app.log          桌面应用运行日志
终端输出               直接跑 python cli.py 时的实时日志
```

界面左侧的「运行日志」面板就是 `app.log` 的实时镜像。

### 常见问题

**ChromaDB 写入报错 `Error loading hnsw index`**

Windows 上 Chroma 偶发只写出部分索引文件。换 Qdrant：

```yaml
vector_store:
  backend: qdrant
```

然后重新建索引。

**显存 OOM / 首轮重排卡住几十秒**

CLIP 和本地 3B 模型抢 8GB 显存。两个办法：

```yaml
embedding:
  device: cpu     # CLIP 回 CPU，查询编码只慢几十毫秒
```

或让 LLM 走 API（默认就是）。

**联网下载 0 张**

Bing 图片搜索页面结构变了，`murl` 字段解析失效。看日志里有没有 `联网搜索没有拿到图片链接`。临时办法是手动下载图片到目录后建索引。

**向量库损坏 / 打不开**

```
Qdrant local: Storage folder is already accessed by another instance
```

说明有旧进程占着锁，任务管理器结束残留的 `python.exe`，或换一个 `path`。

**模型下载失败**

`download_models.py` 默认走 hf-mirror。镜像失效时手动下载：

```
https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16
https://huggingface.co/Salesforce/blip-image-captioning-base
```

放到 `models/` 下对应目录即可，文件名保持不变。

**索引卡在某个百分比不动**

大目录首次索引会跑 BLIP 描述生成，5000 张约 10-20 分钟。看 `logs/app.log` 里的「索引进度 x/y」确认在跑。急着用可以先关描述：

```yaml
caption:
  enabled: false
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
- [x] 229 个单元测试 + CI 配置
- [ ] 混合检索：BM25 + 向量，RRF 融合
- [ ] 中文图像描述（Qwen2.5-VL 替代 BLIP）
- [ ] Recall@K 实测与对比实验
- [ ] 向量库迁移工具（Chroma ↔ Qdrant）
- [ ] 索引备份与快照
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

**检索质量**

- 描述是英文的。BLIP base 只支持英文，中文查询与英文描述之间有语义落差。
- 只做稠密向量召回。对「图中文字」「精确专有名词」「文件编号」支持有限。
- 大批量只重排头部。超过 16 条候选时，只有前 16 条做语义重排，其余按向量分补齐。阈值是 `rerank.max_candidates`。
- 尚未完成 Recall@K 定量评估。

**索引一致性**

- 增量更新靠目录 mtime。修改图片内容但目录 mtime 不变时不会重新索引，需要 `rebuild=True`。
- 移动文件路径后，向量库里的旧路径不会自动清理，会产生孤儿向量。用 `remove_index(missing_only=True, dry_run=False)` 手动清理。
- 支持格式固定为：jpg / jpeg / png / webp / bmp / tif / tiff / gif / avif。HEIC、RAW 等相机格式不支持。
- 以图搜图时，输入图如果不在索引库里，只做向量比对，不会自动入库。

**并发与多会话**

- 索引按文件夹加锁，同一目录不会重复索引。但 `remove_index`、`switch_vector_backend` 没有跨会话锁，多个会话同时操作同一向量库有风险。
- 长期向量记忆按 `user_id` 隔离，当前桌面端只有一个用户，多用户部署需要自行传入不同 user_id。
- 摘要压缩触发条件是 `messages >= max_turns * 2`，目前硬编码在 `run_tasks` 里，没有暴露到配置。

**动态工具**

- 源码黑名单是启发式扫描，不是完整沙箱。已拦截 `subprocess`、`os.system`、`eval`、`open` 等，但无法穷举。
- `Path` 已替换为受限子类，读写删都要过沙箱；`Image.open` 仍未加路径校验，理论上可读任意图片文件。
- 动态工具拿不到 `os.environ`，读不到 API Key。

**联网下载**

- 依赖 Bing 图片搜索页面结构，字段调整会导致解析失效。
- 下载前只校验文件大小（>2KB）和后缀，不校验真实图片格式。恶意文件伪装成 `.jpg` 会写入缓存。

**其他**

- ChromaDB 在部分 Windows 环境写入异常，实测 Qdrant 更稳。
- 8GB 显存同时跑 CLIP 与 3B 模型会触发显存交换，建议 LLM 走 API。
- 导出功能只有 CSV / TXT，不支持直接复制图片文件。

## 测试

```powershell
pytest tests -q
```

229 个测试，全部不依赖网络与模型下载，离线可跑。需要真实模型的场景用 fake pipeline 代替，只有 `test_models` 相关会检查本地文件是否存在，缺失时跳过。

**当前缺的测试类型**：

- 端到端集成测试（自然语言 → 任务拆解 → 工具调用 → 结果）
- 压力测试（上万张图片索引、多会话并发）
- 动态工具对抗测试（黑名单绕过尝试）

## License

MIT。

第三方依赖各自遵循原许可证：

| 依赖 | 许可证 |
| --- | --- |
| PyQt5 | GPL v3 / 商业授权 |
| PyTorch | BSD-3-Clause |
| transformers | Apache-2.0 |
| Chinese-CLIP | MIT |
| BLIP | BSD-3-Clause |
| Qdrant | Apache-2.0 |
| ChromaDB | Apache-2.0 |

**商用注意**：PyQt5 是 GPL v3，闭源商用需购买 Riverbank 商业授权，或改用 PySide6（LGPL）。
