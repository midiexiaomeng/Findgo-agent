# Findgo

**一个会自己判断该做什么的本地图像检索 Agent**

用中文描述找图，能看懂目录、会挑工具、记得你说过的话，找不到就上网补，找到的不重复。

## 这是什么

大多数图片检索工具是这样的：你输入一段文字，它把向量库前 20 条结果吐给你，剩下的靠你自己挑。问题在于，你真正想做的事从来不是「返回 20 条」，而是「帮我在这个文件夹里找 5 张有猫的、不要卡通的图」。

差的那一步，是判断。

Findgo 想做的是把这个判断补上：先看你的话里有没有路径、有没有数量、有没有排除条件，决定该调用哪个工具；检索回来先看成色，置信度太低就换个说法再试一次；本地不够就联网补，补完记得下次别重复返回同一批。整个过程在本地完成，只有可选的语义重排会用到云端大模型。

它现在有 16 个工具、三层记忆、一套路径沙箱，以及一个不占端口、不依赖浏览器的 PyQt5 桌面窗口。说它是「图片检索」，不如说它是「一个只关心图片的助手」。

## 功能特性

**以文搜图** — Chinese-CLIP 把中文和图像映射到同一个语义空间，说「夕阳下的海边」能直接命中，不需要先翻译成英文，也不需要预先打标签。支持限定目录、指定数量、排除已返回的结果、相似图去重。

**以图搜图** — 给一张本地图片找视觉相似的图，也支持直接往窗口里粘贴。输入的图不需要预先入库，当场编码当场比对。

**三级检索** — 跨模态嵌入负责把文字变成向量，向量库负责快速召回，BLIP 生成的图像描述交给大模型做语义重排。大批量检索时只重排头部候选，其余按向量分补齐，保证「找 400 张」不会变成等十分钟。

**双向量库** — ChromaDB 与 Qdrant 共用一套接口，改一行配置就能切换。切换前会做一次自检，失败自动回滚，不会把索引切坏。实测在 Windows 上 Qdrant 更稳。

**联网补图** — 本地检索不够时会去网上找，下载后缓存在独立目录，有容量上限，满了自动淘汰。三种模式：仅本地、混合、仅联网，按钮上直接切。

**三层记忆** — 短期会话记得住上下文，超长对话会压缩成摘要，跨会话的稳定偏好存进向量库。除此之外还有一层结构化事实，是本会话唯一权威的约束源，优先级高于历史对话，防止旧约束盖掉你刚说的话。

**结构化事实增量更新** — 你说「再找三张」，数量从 2 变成 3；你说「不要卡通的」，就多一个过滤条件；你说「清空约束」，就全部重置。事实通过 patch 增量合并，`null` 表示删除某个字段，不需要重写整份状态。

**跨轮排除与复用** — 默认会排除上一轮已经给过你的图，避免每次都返回同一批。但如果你的话里出现「刚才那两张再发一次」，就直接复用，不重新检索，省掉一次完整链路。

**动态工具** — 遇到现有工具覆盖不了的需求，Agent 会现场写一个临时工具：源码先过安全扫描，再在隔离命名空间里编译，最后弹窗让你看代码、决定要不要执行。觉得好用可以一键存进全局工具库。

**安全沙箱** — 所有路径操作走同一个分域沙箱：素材目录只读，工作目录可读写，系统目录一律拒绝，符号链接直接拦。删除走暂存区，先建硬链接再打软删除标记，原图不动；真正删除原图必须经过界面二次确认，Agent 自己无权调用。

**桌面应用** — PyQt5 无边框窗口，自定义标题栏，四边四角可拖拽拉伸。左中右三栏：左侧运行日志与会话列表，中间对话与输入，右侧任务链与文件区。不占端口，不启浏览器，双击 `start-app.bat` 就开。

## 截图

> 主界面截图待补。当前版本可运行 `start-app.bat` 直接查看。

## 前置要求

先确认能跑起来，再往下看。

| 项目 | 最低 | 推荐 |
| --- | --- | --- |
| 系统 | Windows 10 / Linux | Windows 11 |
| Python | 3.9 | 3.11 |
| 内存 | 8 GB | 16 GB |
| 显存 | 可选（可用 CPU） | 6 GB 以上，CLIP 走 GPU |
| 磁盘 | 10 GB | 30 GB（模型 1.6GB + 向量库 + 图片缓存） |

关于「能不能完全离线」：CLIP 与 BLIP 本地跑，首次需联网下载模型（约 1.6GB），之后完全离线。LLM 重排默认走 DeepSeek API，需要 `DEEPSEEK_API_KEY`。想完全离线就把 `config.yaml` 的 `rerank.backend` 改成 `ollama`，本地起一个 Ollama 服务。

没有 API Key 也能用，只是跳过重排，退回纯向量排序，效果会差一些。

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

> **安全警告**：Gradio 默认只监听 `127.0.0.1`，不要改成 `0.0.0.0` 或开 `share=True` 暴露到公网。沙箱是进程内的路径防护，不是网络鉴权；一旦公网可访问，任何人不经认证就能建索引、删文件。

### 命令行用法

```powershell
python cli.py index "D:\Photos"      # 建索引（增量，已索引的跳过）
python cli.py search "夕阳下的海边"   # 文本检索
python cli.py image "D:\test.jpg"    # 以图搜图
python cli.py stats                  # 索引统计
python cli.py doctor                 # 环境自检
python cli.py serve                  # 启动 Gradio
```

### 第一次对话长什么样

```
你：在 D:\val2017 里找 40 张含猫的图片

Agent：
[任务链]
📁 打开 D:\val2017
🔍 40 张含猫的图片

已为你找到 40 张猫的图片。
[前 20 张缩略图，右下角省略号点开完整图库]
```

如果第一次置信度低于阈值，会换个说法再试；点数超过 20 张时气泡右下角出现省略号，点开是完整图库，悬停预览、双击放大。

## 架构

```
src/findgo/
├── agent/            Agent 层：编排、规划、执行、记忆、动态工具
├── retriever/        内核层：CLIP、索引、检索、向量库、重排
├── tools/            16 个工具实现
├── ui/               界面组件（只渲染，不写业务）
├── utils/            沙箱、schema、日志
├── trash.py          暂存区
├── downloads.py      下载缓存
├── long_term.py      Qdrant 长期记忆
├── diagnostics.py    环境自检
└── settings_dialog.py 设置面板

desktop_app.py        启动入口，只做窗口组装
```

分层是这样：最上面是交互层，PyQt5 桌面应用、CLI、可选的 Gradio；往下是 Agent 层，Orchestrator 负责调度，Planner 拆解任务并识别路径，ToolExecutor 统一做参数校验、超时控制、输出截断和权限拦截，Critic 判断结果够不够好，Responder 合成回答，Memory 管理四张表；再往下是工具层，16 个工具共用一套 `BaseTool` 契约；内核层是 Pipeline、Searcher、Indexer、Encoder 和向量库抽象；最底层是沙箱、配置、日志、主题。

不套 LangChain。Planner、Critic、Responder 加起来不到 500 行，逻辑都在明面上，改起来不用先读一遍框架文档。

### 工具清单

| 工具 | 作用 | 权限 |
| --- | --- | --- |
| `image_retrieval` | 文本检索，支持限定目录、数量、排除、去重 | read |
| `image_search_by_image` | 以图搜图 | read |
| `find_duplicates` | 按向量相似度找重复图 | read |
| `filter_results` | 对已有结果二次过滤 | read |
| `describe_image` | 单图 BLIP 描述 | read |
| `folder_scan` | 扫描目录统计图片数 | read |
| `list_indexed_folders` | 列出已索引目录 | read |
| `count_indexed_images` | 索引内数量统计 | read |
| `get_index_stats` | 数量、后端、设备 | read |
| `preview_staged` | 只读查看暂存区清单 | read |
| `index_folder` | 建索引，mtime 缓存 + 文件夹锁 | index_write |
| `remove_index` | 删索引，默认 dry-run，支持清孤儿向量 | index_write |
| `switch_vector_backend` | 切后端，自检失败回滚 | index_write |
| `export_results` | 导出 CSV / TXT | file_write |
| `trash_manage` | 暂存 / 恢复 / 永久删除 / 清理 | file_write |
| `web_image_search` | 联网搜索下载并建索引 | file_write |

权限分四级，前三级 Agent 可以自主调用，第四级 `danger_delete` 会被执行器直接拦下，返回 `need_user_confirm`：

| 等级 | 能做什么 | 谁会触发 |
| --- | --- | --- |
| `read` | 只读文件、查向量库、统计 | 检索、预览、统计类工具 |
| `index_write` | 修改向量库内容 | 建索引、删索引、切后端 |
| `file_write` | 在 `data/` 下写文件 | 导出、暂存、联网下载 |
| `danger_delete` | 删原图 | 仅 `purge`，必须人工确认 |

### 记忆的优先级

```
结构化事实  >  最近对话  >  摘要压缩  >  跨会话向量记忆
```

事实是本会话唯一的权威源。交互式对话每轮自动更新；批处理 `run_tasks` 默认只读不改写，避免多条独立任务互相污染。

```python
memory.patch_facts({"count": 3})
memory.patch_facts({"filters": {"color": None}})   # null 表示删除该字段
memory.clear_facts()
```

### 删除流程

```
stage   → 硬链接进暂存区 + 向量打 is_staged 标记，原图不动   【Agent 自动】
restore → 删链接 + 标记复原                                  【Agent 或用户】
purge   → 删原图 + 删链接 + 向量移除                          【必须人工确认】
cleanup → 按留存天数与容量上限自动清理                        【后台定时 / 手动】
```

Agent 能做的只有 stage 和 restore。purge 是唯一会真正删除原图的操作，被三层拦截：工具层拒绝、执行器拒绝、界面二次确认。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 界面 | PyQt5（无边框窗口 + 自定义 QSS） |
| 跨模态模型 | Chinese-CLIP (`OFA-Sys/chinese-clip-vit-base-patch16`) |
| 图像描述 | BLIP base (`Salesforce/blip-image-captioning-base`) |
| 向量库 | Qdrant local / ChromaDB |
| 推理 | PyTorch 2.7 + CUDA 11.8 |
| LLM | DeepSeek API / Ollama / 本地 Transformers |
| 存储 | SQLite（会话、事实、描述缓存）、Qdrant（向量） |
| 测试 | pytest |

## 平台支持

| 平台 | 状态 |
| --- | --- |
| Windows 10 / 11 | 主要开发与测试环境 |
| Linux | 可用，路径与沙箱逻辑已适配 |
| macOS | 未验证，`sandbox.py` 的路径规则需补充 |
| GPU | CUDA ≥ 11.8；无 GPU 自动回 CPU |

## 开发

```powershell
# 安装开发依赖
pip install -r requirements-dev.txt

# 跑测试（离线可跑，不需要模型和网络）
pytest tests -q

# 环境自检（检查依赖、CUDA、模型、向量库、LLM）
python cli.py doctor

# 代码检查
pre-commit run --all-files
```

新增一个内置工具的话，在 `src/findgo/tools/` 下继承 `BaseTool`，定义 `name`、`description`、`args_schema`、`permission`、`timeout`、`run()`，然后在 `tools/__init__.py` 导出、`orchestrator.py` 的 `self.tools` 注册、`configs/tools.yaml` 登记、补一个测试。

临时需求不必新增工具，直接提给 Agent，它会走动态工具：生成 `DynamicToolSpec`，源码过黑名单扫描，在隔离命名空间编译，弹窗让你审。

## 动态工具怎么用

不必先写代码。直接提一个现有工具盖不住的需求：

```
你：帮我统计这个目录里图片的宽高分布

Agent：现有工具只能数数量，需要临时能力。
[任务链里出现：🔧 image_size_stats｜自定义临时工具｜需要审核]

→ 点「重跑」弹出审核窗口
   看到源码、安全提示
   [拒绝]  [允许执行]
```

只有点了「允许执行」才会编译加载。审核窗口里的「保存到全局工具库」默认不勾，推荐分够高时才预勾选，最终必须你手动确认。

## 记忆在对话里怎么表现

```
你：找两张猫的图片
Agent：找到 2 张。[图片]
       事实：{count: 2}

你：再找三张
Agent：找到 3 张。[另外的图，不重复]
       事实更新：count → 3，上一轮的图被排除

你：不要卡通的
Agent：找到 3 张。[重新检索]
       事实追加：filters.style = 非卡通

你：清空约束
Agent：已重置。
       事实清空：clear_facts()
```

「再找」自动排除上一轮已返回的图；「刚才那两张再发一次」则直接复用上轮结果，不重新检索。

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

优先级：`环境变量 > configs/agent.yaml > 代码默认值`。环境变量格式是 `FINDGO_<SECTION>__<KEY>`，比如 `FINDGO_VECTOR_STORE__QDRANT__PATH` 会覆盖 Qdrant 路径，适合 Docker 或临时调试。

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

Windows 上 Chroma 偶发只写出部分索引文件。把 `config.yaml` 的 `vector_store.backend` 改成 `qdrant`，重新建索引。这个环境实测下来 Qdrant 更可靠。

**显存 OOM / 首轮重排卡住几十秒**

CLIP 和本地 3B 模型抢 8GB 显存。把 `embedding.device` 改成 `cpu`，查询编码只慢几十毫秒，或者让 LLM 走 API（默认就是）。

**联网下载 0 张**

Bing 图片搜索页面结构变了，`murl` 字段解析失效。看日志里有没有「联网搜索没有拿到图片链接」。临时办法是手动下载图片到目录后建索引。

**向量库打不开，报 `Storage folder is already accessed by another instance`**

有旧进程占着锁。任务管理器结束残留的 `python.exe`，或换一个路径。

**模型下载失败**

`download_models.py` 默认走 hf-mirror。镜像失效时手动下载 `OFA-Sys/chinese-clip-vit-base-patch16` 和 `Salesforce/blip-image-captioning-base`，放到 `models/` 下对应目录，文件名保持不变即可。

**索引卡在某个百分比不动**

大目录首次索引会跑 BLIP 描述生成，5000 张约 10-20 分钟。看 `logs/app.log` 里的「索引进度 x/y」确认在跑。急着用可以先把 `caption.enabled` 改成 `false`。

## 已知限制

**检索质量**

- 描述是英文的。BLIP base 只支持英文，中文查询与英文描述之间有语义落差。
- 只做稠密向量召回。对「图中文字」「精确专有名词」「文件编号」支持有限。
- 大批量只重排头部。超过 16 条候选时只有前 16 条做语义重排，其余按向量分补齐，阈值是 `rerank.max_candidates`。
- 尚未完成 Recall@K 定量评估。

**索引一致性**

- 增量更新靠目录 mtime。修改图片内容但目录 mtime 不变时不会重新索引，需要 `rebuild=True`。
- 移动文件路径后，向量库里的旧路径不会自动清理，会产生孤儿向量。用 `remove_index(missing_only=True, dry_run=False)` 手动清理。
- 支持格式固定为 jpg / jpeg / png / webp / bmp / tif / tiff / gif / avif。HEIC、RAW 等相机格式不支持。
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

当前缺的测试类型：端到端集成测试（自然语言 → 任务拆解 → 工具调用 → 结果）、压力测试（上万张图片索引、多会话并发）、动态工具对抗测试（黑名单绕过尝试）。

## 路线图

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
- [x] 界面组件拆分，`desktop_app.py` 只剩组装逻辑
- [ ] 混合检索：BM25 + 向量，RRF 融合
- [ ] 中文图像描述（Qwen2.5-VL 替代 BLIP）
- [ ] Recall@K 实测与对比实验
- [ ] 向量库迁移工具（Chroma ↔ Qdrant）
- [ ] 索引备份与快照
- [ ] PyInstaller 打包 exe

## 提交 Issue

Bug 请附上复现步骤和日志片段（`logs/app.log` 相关行），最好说明你的系统、Python 版本、是否有 GPU。需求请说清想要什么、为什么需要、做到什么程度算完成。

- [提交 Bug](https://github.com/midiexiaomeng/Findgo-agent/issues/new?template=bug_report.md)
- [提交需求](https://github.com/midiexiaomeng/Findgo-agent/issues/new?template=feature_request.md)
- [开发指南](CONTRIBUTING.md)

## 致谢

- 中文跨模态检索基于 [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
- 图像描述基于 [BLIP](https://github.com/salesforce/BLIP)
- 向量检索基于 [Qdrant](https://github.com/qdrant/qdrant)，早期版本用过 ChromaDB
- README 的叙述结构与章节安排参考了 [HanaAgent](https://github.com/liliMozi/openhanako)

## 许可证

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
